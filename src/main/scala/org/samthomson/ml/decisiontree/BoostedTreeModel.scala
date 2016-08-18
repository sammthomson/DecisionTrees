package org.samthomson.ml.decisiontree

import com.typesafe.scalalogging.LazyLogging
import org.samthomson.ml.Weighted
import org.samthomson.ml.WeightedMse.{Stats => MseStats}
import org.samthomson.ml.WeightedMean.{Stats => MeanStats}
import org.samthomson.ml.decisiontree.FeatureSet.Mixed
import org.samthomson.util.StreamFunctions.unfold
import spire.implicits._
import scala.collection.parallel.ParSeq
import scala.math._


@SerialVersionUID(1L)
case class RegressionTreeModel[K, X](feats: Mixed[K, X],
                                     lambda0: Double,
                                     maxDepth: Int) extends LazyLogging {
  val tolerance = 1e-6
  private val am = MseStats.hasAdditiveMonoid[Double]

  def fit(data: Iterable[Weighted[Example[X, Double]]]): (DecisionTree[X, Double], Double) = {
    logger.debug("fitting regression tree, depth: " + maxDepth)
    val mseStats = MseStats.of(data.map(_.map(_.output)))  // TODO: cache
    val baseError = mseStats.error
    lazy val leaf = (Leaf.averaging(data), baseError)
    if (maxDepth <= 1 || data.isEmpty || baseError <= tolerance) {
      leaf
    } else {
      val split = bestSplitAndError(data)
      val (leftData, rightData) = data.partition(e => split(e.input))
      if (leftData.isEmpty || rightData.isEmpty) {
        leaf
      } else {
        val shorterModel: RegressionTreeModel[K, X] = this.copy(maxDepth = maxDepth - 1)
        val (left, leftErr) = shorterModel.fit(leftData)
        val (right, rightErr) = shorterModel.fit(rightData)
        val result = Split(split, left, right)
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

  // finds the split that minimizes squared error
  def bestSplitAndError(examples: Iterable[Weighted[Example[X, Double]]]): Splitter[X] = {
    val allSplits = continuousSplitsAndErrors(examples) ++ binarySplitsAndErrors(examples)
    allSplits.minBy(_._2)._1
  }

  def binarySplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]]): ParSeq[(BoolSplitter[K, X], (Double, Double))] = {
    def stats(xs: Iterable[Weighted[Example[X, Double]]]): MseStats[Double] = MseStats.of(xs.map(_.map(_.output)))
    val binary = feats.binary
    binary.feats.toSeq.par.map(feature => {
      val (l, r) = examples.partition(e => binary.get(e.input)(feature))
      (BoolSplitter(feature)(binary), totalErrAndEvenness(stats(l), stats(r)))
    })
  }

  def continuousSplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]]): ParSeq[(FeatureThreshold[K, X], (Double, Double))] = {
    val continuous = feats.continuous
    continuous.feats.toSeq.par.flatMap(feature => {
      val statsByThreshold =
        examples.groupBy(e => continuous.get(e.input)(feature))
            .mapValues(exs => MseStats.of(exs.map(_.map(_.output))))
            .toVector
            .sortBy(_._1)
      val splits = statsByThreshold.map(_._1).map(v => FeatureThreshold[K, X](feature, v)(continuous))
      val errors = {
        val stats = statsByThreshold.map(_._2)
        // errors of left and right side of each split value.
        // found by taking cumulative stats starting from from left, right, respectively.
        val leftErrors = stats.scanLeft(am.zero)(am.plus).tail
        val rightErrors = stats.scanRight(am.zero)(am.plus).tail
        (leftErrors zip rightErrors).map { case (l, r) => totalErrAndEvenness(l, r) }
      }
      splits zip errors
    })
  }

  def categoricalSplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]],
                                 exclusiveFeats: Set[K]): Seq[(OrSplitter[K, X], (Double, Double))] = {
    val binary = feats.binary
    val stats = exclusiveFeats.par
        .map(feat => feat -> MseStats.of(examples.filter(e => binary.get(e.input)(feat)).map(_.map(_.output))))
        .toVector
        .sortBy(_._2.mean)
    // TODO: consider non-contiguous splits?
    val splits = stats.map(_._1).scanLeft(Set[K]())({ case (s, f) => s + f }).tail.map(s => OrSplitter(s)(binary))
    val errors = {
      val leftErrors = stats.map(_._2).scanLeft(am.zero)(am.plus).tail
      val rightErrors = stats.map(_._2).scanRight(am.zero)(am.plus).tail
      (leftErrors zip rightErrors).map { case (l, r) => totalErrAndEvenness(l, r) }
    }
    splits zip errors
  }

  private def totalErrAndEvenness(l: MseStats[Double],
                                  r: MseStats[Double]): (Double, Double) = {
    (l.error + r.error, abs(l.weight - r.weight))
  }
}

case class BoostedTreeModel[K, X, Y](outputSpace: X => Iterable[Y],
                                     xyFeats: FeatureSet.Mixed[K, (X, Y)],
                                     lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                                     lambda0: Double,
                                     lambda2: Double,
                                     maxDepth: Int) extends LazyLogging {
  private val regression = RegressionTreeModel(xyFeats, lambda0, maxDepth)

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
    val (nextTree, _) = regression.fit(residuals)
    nextTree match {
      case Leaf(avg) if math.abs(avg) <= lambda0 =>
        // don't waste my time with these mickey mouse trees
        logger.debug("Empty tree. Stopping.")
        (None, totalLoss.mean)
      case _ => (Some(nextTree), totalLoss.mean)
    }
  }
}
