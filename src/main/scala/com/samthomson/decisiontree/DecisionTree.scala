package com.samthomson.decisiontree

import com.samthomson.LazyStats.weightedMean
import com.samthomson.Weighted
import com.samthomson.WeightedMse.{Stats => MseStats}
import com.samthomson.decisiontree.FeatureSet.{OneHot, Mixed}
import spire.algebra.Field
import spire.implicits._

import scala.math.{abs, max}


case class Example[+X, +Y](input: X, output: Y)


trait Splitter[-X] extends Function[X, Boolean] {
  def isLeft(input: X): Boolean
  // derived:
  def choose[T](input: X)(left: T, right: T): T = if (isLeft(input)) left else right
  def apply(input: X): Boolean = isLeft(input)
}
case class FeatureThreshold[+F, X](feature: F, threshold: Double)
                                  (implicit cf: FeatureSet.Continuous[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = cf.get(input)(feature) <= threshold
  override def toString: String = s"$feature <= $threshold"
}
case class BoolSplitter[+F, -X](feature: F)(implicit bf: FeatureSet.Binary[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = bf.get(input)(feature)
  override def toString: String = s"$feature"
}
case class OrSplitter[+F, -X](features: Iterable[F])(implicit bf: FeatureSet.Binary[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = features.exists(bf.get(input))
  override def toString: String = s"OR(${features.mkString(", ")})"
}


sealed trait DecisionTree[-X, +Y] extends Model[X, Y] {
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


case class RegressionTree[F, X](feats: Mixed[F, X],
                                lambda0: Double,
                                maxDepth: Int) {
  val tolerance = 1e-6
  // prefer evenly weighted splits (for symmetry breaking)
  val evenWeightPreference = 1e-3

  private val am = MseStats.hasAdditiveMonoid[Double]

  def fit(data: Iterable[Weighted[Example[X, Double]]]): DecisionTree[X, Double] = {
    val mseStats = MseStats.of(data.map(_.map(_.output)))  // TODO: cache
    val baseError = mseStats.error
    if (maxDepth <= 1 || data.isEmpty || baseError <= tolerance) {
      Leaf.averaging(data)
    } else {
      val (split, error) = bestSplitAndError(data)
      val (leftData, rightData) = data.partition(e => split(e.input))
      // TODO: Better to prune afterwards than to stop early. Sometimes you need to make splits
      // that don't improve in order to make later splits that do improve.
      if (leftData.isEmpty || rightData.isEmpty || baseError - error + tolerance < lambda0) {
        Leaf.averaging(data)
      } else {
        val shorter: RegressionTree[F, X] = this.copy(maxDepth = maxDepth - 1)
        val left = shorter.fit(leftData)
        val right = shorter.fit(rightData)
        Split(split, left, right)
      }
    }
  }

  // finds the split that minimizes squared error
  def bestSplitAndError(examples: Iterable[Weighted[Example[X, Double]]]): (Splitter[X], Double) = {
    val allSplits = continuousSplitsAndErrors(examples) ++ binarySplitsAndErrors(examples)
    val (split, (err, _)) = allSplits.minBy({ case (s, (er, even)) => er + evenWeightPreference * even })
    (split, err)
  }

  def binarySplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]]): Seq[(BoolSplitter[F, X], (Double, Double))] = {
    val binary = feats.binary
    binary.feats.toSeq.map(feature => {
      val stats = examples.groupBy(e => binary.get(e.input)(feature))
          .mapValues(exs => MseStats.of(exs.map(_.map(_.output))))
          .withDefaultValue(am.zero)
      val l = stats(true)
      val r = stats(false)
      (BoolSplitter(feature)(binary), totalErrAndEvenness(l, r))
    })
  }

  def continuousSplitsAndErrors(examples: Iterable[Weighted[Example[X, Double]]]): Seq[(FeatureThreshold[F, X], (Double, Double))] = {
    val continuous = feats.continuous
    continuous.feats.toSeq.flatMap(feature => {
      val statsByThreshold =
        examples.groupBy(e => continuous.get(e.input)(feature))
            .mapValues(exs => MseStats.of(exs.map(_.map(_.output))))
            .toVector
            .sortBy(_._1)
      val splits = statsByThreshold.map(_._1).map(v => FeatureThreshold[F, X](feature, v)(continuous))
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
                                 exclusiveFeats: Set[F]): Seq[(OrSplitter[F, X], (Double, Double))] = {
    val binary = feats.binary
    val stats = exclusiveFeats
        .map(feat => feat -> MseStats.of(examples.filter(e => binary.get(e.input)(feat)).map(_.map(_.output))))
        .toVector
        .sortBy(_._2.mean)
    val splits = stats.map(_._1).scanLeft(Set[F]())({ case (s, f) => s + f }).tail.map(s => OrSplitter(s)(binary))
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
