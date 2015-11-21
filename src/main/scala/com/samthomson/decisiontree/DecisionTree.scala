package com.samthomson.decisiontree

import com.samthomson.LazyStats.weightedMean
import com.samthomson.Weighted
import com.samthomson.WeightedMse.{Stats => MseStats}
import spire.algebra.Field
import spire.implicits._

import scala.math.max


case class Example[+X, +Y](input: X, output: Y)


trait Splitter[-X] extends Function[X, Boolean] {
  def choose[T](input: X)(left: T, right: T): T = if (apply(input)) left else right
}
case class FeatureThreshold[+F, X](feature: F, threshold: Double)
                                  (implicit CF: FeatureSet.Continuous[F, X]) extends Splitter[X] {
  override def apply(input: X): Boolean = CF.get(input)(feature) <= threshold
  override def toString: String = s"$feature <= $threshold"
}
case class BoolSplitter[+F, -X](feature: F)(implicit BF: FeatureSet.Binary[F, X]) extends Splitter[X] {
  override def apply(input: X): Boolean = BF.get(input)(feature)
  override def toString: String = s"$feature"
}

sealed trait DecisionTree[-X, +Y] {
  def depth: Int
  def predict(input: X): Y
  def prettyPrint(indent: String = ""): String
  override def toString: String = prettyPrint()
}
case class Split[-X, Y](split: Splitter[X],
                        left: DecisionTree[X, Y],
                        right: DecisionTree[X, Y]) extends DecisionTree[X, Y] {
  override val depth = max(left.depth, right.depth) + 1
  final override def predict(input: X): Y = split.choose(input)(left, right).predict(input)
  override def prettyPrint(indent: String): String = {
    val indented = indent + "  "
    indent + s"Split(\n" +
        indented + split + "\n" +
        left.prettyPrint(indented) + ",\n" +
        right.prettyPrint(indented) + "\n" +
    indent + ")"
  }
}
case class Leaf[-X, Y](constant: Y) extends DecisionTree[X, Y] {
  override val depth = 1
  final override def predict(input: X): Y = constant
  override def prettyPrint(indent: String): String = indent + s"Leaf($constant)"
}
object Leaf {
  def averaging[X, Y: Field](examples: Iterable[Weighted[Example[X, Y]]]): Leaf[X, Y] = {
    val weightedOutputs = examples.map(_.map(_.output))
    Leaf(weightedMean(weightedOutputs))
  }
}

object RegressionTree {
  import FeatureSet.Mixed
  val tolerance = 1e-7
  val evenWeightPreference = 0.1

  def fit[F, X](data: Iterable[Weighted[Example[X, Double]]],
                maxDepth: Int)
               (implicit mf: Mixed[F, X]): DecisionTree[X, Double] = {
    val mseStats = MseStats.of(data.map(_.map(_.output)))
    if (maxDepth <= 1 || data.isEmpty || mseStats.meanSquaredError <= tolerance) {
      Leaf.averaging(data)
    } else {
      val (split, _) = bestSplitAndError(data)
      val (leftData, rightData) = data.partition(e => split(e.input))
      if (leftData.isEmpty || rightData.isEmpty) {
        Leaf.averaging(data)
      } else {
        val left = fit(leftData, maxDepth - 1)
        val right = fit(rightData, maxDepth - 1)
        Split(split, left, right)
      }
    }
  }

  // finds the split that minimizes squared error
  private def bestSplitAndError[F, X](examples: Iterable[Weighted[Example[X, Double]]])
                                     (implicit m: Mixed[F, X]): (Splitter[X], Double) = {
    val am = MseStats.hasAdditiveMonoid[Double]
    implicit val binary = m.binary
    implicit val continuous = m.continuous
    // TODO: cache
    val idealWeight = examples.map(_.weight).sum / 2.0

    val continuousSplitsAndErrs = m.continuous.feats.toSeq.flatMap(feature => {
      val statsByThreshold =
          examples.groupBy(e => m.continuous.get(e.input)(feature))
              .mapValues(exs => MseStats.of(exs.map(_.map(_.output)))).toVector.sortBy(_._1)
      val splits = statsByThreshold.map(_._1).map(v => FeatureThreshold[F, X](feature, v))
      val errors = {
        // errors of left and right side of each split value
        val leftErrors = statsByThreshold.map(_._2).scanLeft(am.zero)(am.plus).tail
        val rightErrors = statsByThreshold.map(_._2).scanRight(am.zero)(am.plus).tail
        for ((l, r) <- leftErrors zip rightErrors) yield {
          l.weight * l.meanSquaredError + r.weight * r.meanSquaredError +
              evenWeightPreference * math.abs(l.weight - idealWeight)
        }
      }
      splits zip errors
    })
    val binarySplitsAndErrs = m.binary.feats.toSeq.map(feature => {
      val stats = examples.groupBy(e => m.binary.get(e.input)(feature))
                    .mapValues(exs => MseStats.of(exs.map(_.map(_.output))))
      val l = stats.getOrElse(true, am.zero)
      val r = stats.getOrElse(false, am.zero)
      val error =
        l.weight * l.meanSquaredError + r.weight * r.meanSquaredError +
            evenWeightPreference * math.abs(l.weight - idealWeight)
      (BoolSplitter(feature), error)
    })
    // break ties by most evenly weighted split
    (continuousSplitsAndErrs ++ binarySplitsAndErrs).minBy(_._2)
  }
}
