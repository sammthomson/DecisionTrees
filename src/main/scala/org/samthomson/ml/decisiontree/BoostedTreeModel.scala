package org.samthomson.ml.decisiontree


import com.typesafe.scalalogging.LazyLogging
import org.samthomson.ml.{WeightedMse, Weighted}
import org.samthomson.ml.WeightedMse.{Stats => MseStats}
import org.samthomson.ml.WeightedMean.{Stats => MeanStats}
import org.samthomson.ml.decisiontree.FeatureSet.Mixed
import org.samthomson.util.StreamFunctions.unfold
import spire.implicits._
import scala.collection.{mutable => m, GenMap, GenSeq}
import scala.math._


/**
  * A column-indexed database of training examples
  *
  * @param inputs the original examples
  * @param binaryIndex map from feature to set of idxs of examples for which that feature fires
  * @param continuousIndex map from feature to iterable of (feature value, example idx) pairs
  * @tparam X the input type
  * @tparam Y the output type
  * @tparam K the type of features
  */
case class IndexedExamples[X, Y, K](inputs: IndexedSeq[X],
                                    outputs: IndexedSeq[Weighted[Y]],
                                    binaryIndex: GenMap[K, Set[Int]],
                                    continuousIndex: GenMap[K, GenSeq[(Double, Set[Int])]],
                                    categoricalIndex: GenMap[K, GenMap[String, Set[Int]]]) {
  def example(i: Int): Weighted[Example[X, Y]] = outputs(i).map(Example(inputs(i), _))
}
object IndexedExamples extends LazyLogging {
  def build[X, Y, K](data: TraversableOnce[Weighted[Example[X, Y]]])
                    (implicit mixed: Mixed[K, X]): IndexedExamples[X, Y, K] = {
    logger.debug("Building feature index.")
    val examples = data.toVector
    val inputs = examples.map(_.input)
    val outputs = examples.map(_.map(_.output))
    val binary = mixed.binary
    val continuous = mixed.continuous
    val categorical = mixed.categorical
    val binaryIndex = m.Map.empty[K, m.Set[Int]]
    val continuousIndex = m.Map.empty[K, m.Map[Double, m.Set[Int]]]
    val categoricalIndex = m.Map.empty[K, m.Map[String, m.Set[Int]]]
    examples.zipWithIndex.foreach {
      case (Weighted(Example(x, y), w), i) =>
        // TODO: use DefaultDict
        for ((k, v) <- binary.featVals(x)) {
          if (!binaryIndex.contains(k)) binaryIndex(k) = m.Set.empty
          binaryIndex(k) += i
        }
        for ((k, v) <- continuous.featVals(x)) {
          if (!continuousIndex.contains(k)) continuousIndex(k) = m.Map.empty
          if (!continuousIndex(k).contains(v)) continuousIndex(k)(v) = m.Set.empty
          continuousIndex(k)(v) += i
        }
        for ((k, v) <- categorical.featVals(x)) {
          if (!categoricalIndex.contains(k)) categoricalIndex(k) = m.Map.empty
          if (!categoricalIndex(k).contains(v)) categoricalIndex(k)(v) = m.Set.empty
          categoricalIndex(k)(v) += i
        }
    }
    logger.debug("Done building feature index.")
    IndexedExamples(
      inputs,
      outputs,
      Map() ++ binaryIndex.mapValues(_.toSet),
      Map() ++ continuousIndex.mapValues(_.mapValues(_.toSet).toSeq.sortBy(_._1)),
      Map() ++ categoricalIndex.mapValues(_.mapValues(_.toSet))
    )
  }
}

@SerialVersionUID(1L)
case class RegressionTreeModel[K, X](feats: Mixed[K, X],
                                     lambda0: Double,
                                     maxDepth: Int) extends LazyLogging {
  import RegressionTreeModel.LeftAndRightStats

  val tolerance = 1e-6
  private val G = MseStats.hasAdditiveGroup[Double]

  def fit(db: IndexedExamples[X, Double, K],
          indices: Set[Int], // which of all the examples we're actually fitting to
          baseStats: WeightedMse.Stats[Double]): (DecisionTree[X, Double], Double) = {
    logger.debug("fitting regression tree, depth: " + maxDepth)
    lazy val leaf = (Leaf(baseStats.mean), baseStats.error)
    if (maxDepth <= 1 || indices.isEmpty || baseStats.error <= tolerance) {
      leaf
    } else {
      bestSplitAndStats(db, indices, baseStats) match {
        case None => leaf
        case Some((split, errs)) =>
          logger.debug(s"found best split: $split, $errs")
          // TODO: I think it would be faster to split/partition the db also
          val (leftData, rightData) = indices.partition(i => split(db.inputs(i)))
          val shorterModel: RegressionTreeModel[K, X] = this.copy(maxDepth = maxDepth - 1)
          val (leftTree, leftStats) = shorterModel.fit(db, leftData, errs.left)
          val (rightTree, rightStats) = shorterModel.fit(db, rightData, errs.right)
          val result = Split(split, leftTree, rightTree)
          val newStats = leftStats + rightStats
          if (baseStats.error - newStats + tolerance < lambda0 * (result.numNodes - 1)) {
            logger.debug(s"pruning: $result")
            leaf
          } else {
            (result, newStats)
          }
      }
    }
  }

  def fit(data: Iterable[Weighted[Example[X, Double]]]): (DecisionTree[X, Double], Double) = {
    // TODO: when boosting, the db stays the same except for outputs. no need to build the whole thing again.
    val db = IndexedExamples.build(data)(feats)
    fit(db, db.inputs.indices.toSet, MseStats.of(data.map(_.map(_.output))))
  }

  private def stats(db: IndexedExamples[X, Double, K])(xs: TraversableOnce[Int]): MseStats[Double] =
    MseStats.of(xs.map(db.example).map(_.map(_.output)))

  // finds the split that minimizes squared error
  def bestSplitAndStats(examples: IndexedExamples[X, Double, K],
                        indices: Set[Int],
                        totalStats: MseStats[Double]): Option[(Splitter[X], LeftAndRightStats)] = {
    val allSplits =
      continuousSplitsAndStats(examples, indices) ++
          binarySplitsAndStats(examples, indices, totalStats) ++
          categoricalSplitsAndErrors(examples, indices, totalStats)
    val nonUselessSplits = allSplits.filter { case (split, stats) => stats.left.weight > 0 && stats.right.weight > 0 }
    if (nonUselessSplits.nonEmpty) {
      Some(nonUselessSplits.minBy(_._2.totalErrAndEvenness))
    } else {
      None
    }
  }

  def binarySplitsAndStats(db: IndexedExamples[X, Double, K],
                           indices: Set[Int],
                           totalStats: WeightedMse.Stats[Double]): GenMap[BoolSplitter[K, X], LeftAndRightStats] =
    db.binaryIndex.par.map { case (k, xs) =>
      val leftIdxs = xs.intersect(indices)
      val leftStats = stats(db)(leftIdxs)
      val rightStats = totalStats - leftStats
      (BoolSplitter(k)(feats.binary), LeftAndRightStats(leftStats, rightStats))
    }

  def continuousSplitsAndStats(db: IndexedExamples[X, Double, K],
                               indices: Set[Int]): GenMap[FeatureThreshold[K, X], LeftAndRightStats] =
    db.continuousIndex.par.flatMap { case (k, thresholds) =>
      val statsByThreshold = thresholds.map({ case (thresh, exs) =>
        (thresh, MseStats.of(exs.intersect(indices).map(db.example).map(_.map(_.output))))
      })
      val splits = thresholds.map(_._1).map(v => FeatureThreshold[K, X](k, v)(feats.continuous))
      val errors = {
        val stats = statsByThreshold.map(_._2)
        // errors of left and right side of each split value.
        // found by taking cumulative stats starting from from left, right, respectively.
        val leftStats = stats.scanLeft(G.zero)(G.plus).tail
        val rightStats = stats.scanRight(G.zero)(G.plus).tail
        (leftStats zip rightStats).map { case (l, r) => LeftAndRightStats(l, r) }
      }
      splits zip errors
    }

  def categoricalSplitsAndErrors(db: IndexedExamples[X, Double, K],
                                 indices: Set[Int],
                                 totalStats: MseStats[Double]): GenMap[OrSplitter[K, X], LeftAndRightStats] = {
    implicit val categorical = feats.categorical
    db.categoricalIndex.par.flatMap { case (k, valMap) =>
      // for each feature, group examples by value, and sort by their mean
      val statsByVal = valMap.mapValues({ xs =>
        stats(db)(xs.intersect(indices))
      }).filter(_._2.weight > 0.0).toVector.sortBy(_._2.mean)
      // consider each split point along the sorted list
      statsByVal.scanLeft((OrSplitter(k, Set()), LeftAndRightStats(MseStats.of(Seq()), totalStats))) {
        case ((acc, accStats), (v, kStats)) =>
          OrSplitter(k, acc.values + v) ->
              LeftAndRightStats(accStats.left + kStats, accStats.right - kStats)
      }
    }
  }
}

object RegressionTreeModel {
  /** Weight, mean, and variance for both sides of a split of the data */
  case class LeftAndRightStats(left: MseStats[Double],
                               right: MseStats[Double]) {
    def totalErrAndEvenness: ErrorAndEvenness =
      ErrorAndEvenness(left.error + right.error, abs(left.weight - right.weight))
  }
  /** How we rank splits: first by error, then evenness to break ties */
  case class ErrorAndEvenness(error: Double, evenness: Double)
  object ErrorAndEvenness {
    implicit val ord: Ordering[ErrorAndEvenness] = Ordering.by(x => (x.error, x.evenness))
  }
}

case class BoostedTreeModel[K, X, Y](outputSpace: X => Iterable[Y],
                                     xyFeats: FeatureSet.Mixed[K, (X, Y)],
                                     lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                                     lambda0: Double,
                                     lambda2: Double,
                                     maxDepth: Int) extends LazyLogging {
  private val regressionModel = RegressionTreeModel(xyFeats, lambda0, maxDepth)

  def toModel(forest: Vector[Model[(X, Y), Double]]) = MultiClassModel[X, Y](outputSpace, Ensemble(forest))

  def fit(data: Iterable[Example[X, Y]],
          numIterations: Int): (MultiClassModel[X, Y], Double) = {
    val (lastModel, lastScore) = optimizationPath(data, Ensemble(Vector[Model[(X, Y), Double]]())).take(numIterations).last
    (MultiClassModel(outputSpace, lastModel), lastScore)
  }

  def optimizationPath(examples: Iterable[Example[X, Y]],
                       initialModel: Model[(X, Y), Double]): Stream[(Ensemble[(X, Y), Double], Double)] = {
    // cache outputSpaces to make sure candidates always appear in same order
    val examplesAndCandidates = examples.map { ex => (ex, outputSpace(ex.input).toVector) }.toVector
    val db = {
      val ioPairs = examplesAndCandidates.flatMap { case (e, candidates) =>
        // example output will get replaced inside `fitNextTree
        candidates.map(y => Weighted(Example((e.input, y), 0.0), 1.0))
      }
      IndexedExamples.build(ioPairs)(xyFeats)
    }
    // cache the scores from previous iteration, so training runtime doesn't increase as the ensemble grows
    val initialScores = {
      val zeroScores = Map.empty[Y, Double].withDefaultValue(0.0)
      Vector.fill(examplesAndCandidates.size)(zeroScores).par
    }
    unfold((Vector(initialModel), initialScores)) { case (forest, allOldScores) =>
      val modelDiff = toModel(Vector(forest.last))
      logger.debug("calculating scores")
      val allScores = examplesAndCandidates.par.zip(allOldScores).map { case ((e, _), oldScores) =>
        // our forest-so-far produces a score (real number) for each (input, output) pair.
        oldScores + modelDiff.scores(e.input).toMap
      }
      logger.debug("calculating residuals")
      val (residuals, loss) = getResiduals(examplesAndCandidates.zip(allScores))
      val oTree = fitNextTree(db, residuals)
      oTree.map({ tree =>
        // yield the new tree, and update the "unfold" state to include the new tree
        val newForest = forest :+ tree
        ((Ensemble(newForest), loss), (newForest, allScores))
      })
    }
  }

  def getResiduals(examplesCandidatesAndScores: Vector[((Example[X, Y], Vector[Y]), Map[Y, Double])]): (Vector[Weighted[Double]], Double) = {
    // Calculate a 2nd-order Taylor expansion of loss w.r.t. scores, then
    // do the old AdaBoost thing where you turn the quadratic fns into a weighted
    // regression problem.
    var totalLoss = MeanStats(0.0, 0.0) // keep track of objective value (loss)
    var n = 0
    // number of examples
    val residuals = examplesCandidatesAndScores.flatMap { case ((e, candidates), scores) =>
      // calculate loss and its first two derivatives with respect to scores.
      // hessian is actually a diagonal approximation (i.e. ignores interactions btwn scores).
      val (loss, gradient, hessian) = lossFn.lossGradAndHessian(e.output)(scores)
      totalLoss += MeanStats(1.0, loss)
      n += 1
      candidates.map { y =>
        val grad = gradient(y)
        val hess = hessian(y) + lambda2
        // newLoss ~= w * (.5 * hess * theta^2 + grad * theta + oldLoss)    // 2nd-order Taylor approx
        //          = .5 * w * hess * (-grad/hess - theta)^2 + Constant
        //          = weighted squared error of theta w.r.t. -grad / hess
        val residual = -grad / hess
        val weight = .5 * hess
        Weighted(residual, weight)
      }
    }
    logger.debug(f"loss per example: ${totalLoss.mean}%10.5f")
    (residuals, totalLoss.mean)
  }

  def fitNextTree(db: IndexedExamples[(X, Y), Double, K],
                  residuals: Vector[Weighted[Double]]): Option[DecisionTree[(X, Y), Double]] = {
    val (nextTree, _) = regressionModel.fit(
      db.copy(outputs = residuals),
      db.inputs.indices.toSet,
      MseStats.of(residuals)
    )
    nextTree match {
      case Leaf(avg) if math.abs(avg) <= lambda0 =>
        // don't waste my time with these mickey mouse trees
        logger.debug("Empty tree. Stopping.")
        None
      case _ => Some(nextTree)
    }
  }
}
