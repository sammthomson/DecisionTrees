package org.samthomson.ml.decisiontree

import com.typesafe.scalalogging.LazyLogging
import org.samthomson.ml.Weighted
import org.samthomson.ml.WeightedMse.{Stats => MseStats}
import org.samthomson.ml.WeightedMean.{Stats => MeanStats}
import org.samthomson.ml.decisiontree.FeatureSet.Mixed
import org.samthomson.util.StreamFunctions.unfold
import spire.implicits._
import scala.collection.{GenMap, GenSeq}
import scala.math._


/**
  * A column-indexed database of training examples
 *
  * @param examples the original examples
  * @param binaryIndex map from feature to set of idxs of examples for which that feature fires
  * @param continuousIndex map from feature to iterable of (feature value, example idx) pairs
  * @tparam X the input type
  * @tparam Y the output type
  * @tparam K the type of features
  */
case class IndexedExamples[X, Y, K](examples: IndexedSeq[Weighted[Example[X, Y]]],
                                    binaryIndex: GenMap[K, Set[Int]],
                                    continuousIndex: GenMap[K, GenSeq[(Double, Set[Int])]])
object IndexedExamples {
  def build[X, Y, K](data: TraversableOnce[Weighted[Example[X, Y]]])
                    (implicit mixed: Mixed[K, X]): IndexedExamples[X, Y, K] = {
    val examples = data.toVector
    val binaryIndex = Map() ++ mixed.binary.feats.map { k =>
      val getK = mixed.binary.get(k)(_)
      k -> examples.zipWithIndex.collect { case (x, i) if getK(x.input) => i }.toSet
    }
    val continuousIndex = Map() ++ mixed.continuous.feats.toSeq.par.map(k => {
      val getK = mixed.continuous.get(k)(_)
      val exampleIdxsByThreshold =
        examples.zipWithIndex.groupBy({ case (x, i) => getK(x.input) })
            .mapValues(_.map(_._2).toSet)
            .toVector
            .sortBy(_._1)
      k -> exampleIdxsByThreshold
    })
    IndexedExamples(examples, binaryIndex, continuousIndex)
  }
}

@SerialVersionUID(1L)
case class RegressionTreeModel[K, X](feats: Mixed[K, X],
                                     lambda0: Double,
                                     maxDepth: Int) extends LazyLogging {
  import RegressionTreeModel.LeftAndRightStats

  val tolerance = 1e-6
  private val am = MseStats.hasAdditiveMonoid[Double]

  def fit(db: IndexedExamples[X, Double, K],
          indices: Set[Int], // which of all the examples we're actually fitting to
          baseError: Double): (DecisionTree[X, Double], Double) = {
    logger.debug("fitting regression tree, depth: " + maxDepth)
    lazy val leaf = (Leaf.averaging(indices.map(db.examples)), baseError)
    if (maxDepth <= 1 || indices.isEmpty || baseError <= tolerance) {
      leaf
    } else {
      bestSplitIdxsAndStats(db, indices) match {
        case None => leaf
        case Some((split, errs)) =>
          // TODO: I think it would be faster to split/partition the db also
          val (leftData, rightData) = indices.partition(i => split(db.examples(i).input))
          val shorterModel: RegressionTreeModel[K, X] = this.copy(maxDepth = maxDepth - 1)
          val (leftTree, leftErr) = shorterModel.fit(db, leftData, errs.left.error)
          val (rightTree, rightErr) = shorterModel.fit(db, rightData, errs.right.error)
          val result = Split(split, leftTree, rightTree)
          val error = leftErr + rightErr
          if (baseError - error + tolerance < lambda0 * (result.numNodes - 1)) {
            logger.debug(s"pruning: $result")
            leaf
          } else {
            (result, error)
          }
      }
    }
  }

  def fit(data: Iterable[Weighted[Example[X, Double]]]): (DecisionTree[X, Double], Double) = {
    // TODO: when boosting, the db stays the same except for outputs. no need to build the whole thing again.
    val db = IndexedExamples.build(data)(feats)
    fit(db, db.examples.indices.toSet, MseStats.of(data.map(_.map(_.output))).error)
  }

  private def splitData(indices: Set[Int], xs: Set[Int]): (Set[Int], Set[Int]) =
    (xs.intersect(indices), indices.diff(xs))

  private def stats(db: IndexedExamples[X, Double, K])(xs: TraversableOnce[Int]): MseStats[Double] =
    MseStats.of(xs.map(db.examples).map(_.map(_.output)))

  // finds the split that minimizes squared error
  def bestSplitIdxsAndStats(examples: IndexedExamples[X, Double, K],
                            indices: Set[Int]): Option[(Splitter[X], LeftAndRightStats)] = {
    val allSplits = continuousSplitsAndStats(examples, indices) ++ binarySplitsAndStats(examples, indices)
    val nonUselessSplits = allSplits.filter { case (split, stats) => stats.left.weight > 0 && stats.right.weight > 0 }
    if (nonUselessSplits.nonEmpty) {
      Some(nonUselessSplits.minBy(_._2.totalErrAndEvenness))
    } else {
      None
    }
  }

  def binarySplitsAndStats(db: IndexedExamples[X, Double, K],
                           indices: Set[Int]): GenMap[BoolSplitter[K, X], LeftAndRightStats] =
    db.binaryIndex.par.map { case (k, xs) =>
      val (l, r) = splitData(indices, xs)
      (BoolSplitter(k)(feats.binary), LeftAndRightStats(stats(db)(l), stats(db)(r)))
    }

  def continuousSplitsAndStats(db: IndexedExamples[X, Double, K],
                               indices: Set[Int]): GenMap[FeatureThreshold[K, X], LeftAndRightStats] =
    db.continuousIndex.par.flatMap { case (k, thresholds) =>
      val statsByThreshold = thresholds.map({ case (thresh, exs) =>
        (thresh, MseStats.of(exs.intersect(indices).map(db.examples).map(_.map(_.output))))
      })
      val splits = thresholds.map(_._1).map(v => FeatureThreshold[K, X](k, v)(feats.continuous))
      val errors = {
        val stats = statsByThreshold.map(_._2)
        // errors of left and right side of each split value.
        // found by taking cumulative stats starting from from left, right, respectively.
        val leftStats = stats.scanLeft(am.zero)(am.plus).tail
        val rightStats = stats.scanRight(am.zero)(am.plus).tail
        (leftStats zip rightStats).map { case (l, r) => LeftAndRightStats(l, r) }
      }
      splits zip errors
    }

  def categoricalSplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]],
                                 exclusiveFeats: Set[K]): Seq[(OrSplitter[K, X], LeftAndRightStats)] = {
    // TODO: use IndexedExamples
    val binary = feats.binary
    val stats = exclusiveFeats.par
        .map({feat =>
          val getK = binary.get(feat)(_)
          feat -> MseStats.of(examples.filter(e => getK(e.input)).map(_.map(_.output)))
        }).toVector
        .sortBy(_._2.mean)
    // TODO: consider non-contiguous splits?
    val splits = stats.map(_._1).scanLeft(Set[K]())({ case (s, f) => s + f }).tail.map(s => OrSplitter(s)(binary))
    val errors = {
      val leftErrors = stats.map(_._2).scanLeft(am.zero)(am.plus).tail
      val rightErrors = stats.map(_._2).scanRight(am.zero)(am.plus).tail
      (leftErrors zip rightErrors).map { case (l, r) => LeftAndRightStats(l, r) }
    }
    splits zip errors
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
    val (lastModel, lastScore) = optimizationPath(data)(Ensemble(Vector[Model[(X, Y), Double]]())).take(numIterations).last
    (MultiClassModel(outputSpace, lastModel), lastScore)
  }

  def optimizationPath(data: Iterable[Example[X, Y]])
                      (initialModel: Model[(X, Y), Double]): Stream[(Ensemble[(X, Y), Double], Double)] = {
    unfold (Vector(initialModel)) { forest =>
      val (oTree, loss) = fitNextTree(data, toModel(forest))
      oTree.map({ tree =>
        // yield the new tree, and update the "unfold" state to include the new tree
        val newForest = forest :+ tree
        ((Ensemble(newForest), loss), newForest)
      })
    }
  }

  def fitNextTree(data: Iterable[Example[X, Y]],
                  currentModel: MultiClassModel[X, Y]): (Option[DecisionTree[(X, Y), Double]], Double) = {
    var totalLoss = MeanStats(0.0, 0.0)  // keep track of objective value (loss)
    var n = 0  // number of examples
    // Calculate a 2nd-order Taylor expansion of loss w.r.t. scores, then
    // do the old AdaBoost thing where you turn the quadratic fns into a weighted
    // regression problem.
    logger.debug("calculating residuals")
    val residuals = data.flatMap { case Example(input, goldLabel) =>
      // our forest-so-far produces a score (real number) for each (input, output) pair.
      val scores = currentModel.scores(input).toMap
      // calculate loss and its first two derivatives with respect to scores.
      // hessian is actually a diagonal approximation (i.e. ignores interactions btwn scores).
      val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
      totalLoss += MeanStats(1.0, loss)
      n += 1
      gradient.map { case (y, grad) =>
        val hess = hessian(y) + lambda2
        // newLoss ~= w * (.5 * hess * theta^2 + grad * theta + oldLoss)    // 2nd-order Taylor approx
        //          = .5 * w * hess * (-grad/hess - theta)^2 + Constant
        //          = weighted squared error of theta w.r.t. -grad / hess
        val argmin = -grad / hess
        val weight = .5 * hess
        Weighted(Example((input, y), argmin), weight)
      }
    }
    logger.debug(f"loss per example: ${totalLoss.mean}%10.5f")
    val (nextTree, _) = regressionModel.fit(residuals)
    nextTree match {
      case Leaf(avg) if math.abs(avg) <= lambda0 =>
        // don't waste my time with these mickey mouse trees
        logger.debug("Empty tree. Stopping.")
        (None, totalLoss.mean)
      case _ => (Some(nextTree), totalLoss.mean)
    }
  }
}
