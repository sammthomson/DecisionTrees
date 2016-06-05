package org.samthomson.ml.decisiontree

import io.circe.syntax._
import org.samthomson.ml.Weighted
import org.samthomson.ml.decisiontree.FeatureSet.OneHot
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{FlatSpec, Matchers}

object RegressionTreeTest {
  val continuousFeats = Set("a", "b")
  val abFeats = MixedMap.feats(Set[String](), continuousFeats)
  private val toExample: (((Double, Double), Double)) => Example[MixedMap[String], Double] = {
    case ((a, b), y) => Example(MixedMap(Set(), Map("a" -> a, "b" -> b)), y)
  }
  val lambda0 = 0.1
  val data = Vector(
    ((-1.0, 1.0), -1.0),
    (( 0.0, 1.0),  1.0),
    (( 1.0, 1.0),  5.0),
    (( 1.1, 1.0),  4.0)
  ).map(toExample).map(Weighted(_, 1.0))
}

class RegressionTreeTest extends FlatSpec with Matchers with GeneratorDrivenPropertyChecks {
  import RegressionTreeTest._
  val regressionModel = RegressionTree(abFeats, lambda0, maxDepth = 3)

  "RegressionTree.fitRegression" should "fit perfectly given enough depth" in {
    val tree = regressionModel.fit(data)
    tree.depth should be (3)
    for (d <- data) {
      tree.predict(d.input) should be (d.output)
    }
  }

  it should "respect maxDepth" in {
    forAll { (rawData: Vector[(((Double, Double), Double), Double)]) =>
      val data = rawData.map({ case (e, w) => Weighted(toExample(e), math.abs(w)) })
      val tree = regressionModel.copy(maxDepth = 2).fit(data)
      tree.depth should be <= 2
    }
  }

  it should "fit approximately given a little depth" in {
    val tree = regressionModel.copy(maxDepth = 2).fit(data)
    for (d <- data) {
      tree.predict(d.input) should be (d.output +- 1.0)
    }
  }

  it should "stop when there is 0 error" in {
    val constantData = data.map(_.map(_.copy(output = 2.3)))
    val tree = regressionModel.copy(maxDepth = 5).fit(constantData)
    tree.depth should be (1)
    for (d <- constantData) {
      tree.predict(d.input) should be (d.output)
    }
  }

  "categoricalSplitsAndErrors" should "find subsets of features with equal weights" in {
    val data = Vector(
      (("a", -1.0), 1.0),
      (("b",  1.0), 1.0),
      (("c",  4.0), 1.0),
      (("d",  5.0), 1.0)
    ).map({ case ((s, y), w) => Weighted(Example(s, y), w) })
    val feats = Set("a", "b", "c", "d")
    val model = RegressionTree(OneHot(feats), lambda0, maxDepth = 1)
    val splits = model.categoricalSplitsAndErrors(data, feats)
    val (bestSplit, _) = splits.minBy(_._2._1)
    val expected = Set("a", "b")
    bestSplit.features.toSet should be (expected)
  }

  it should "find subsets of features with unequal weights" in {
    val data = Vector(
      (("a", -1.0), 0.01),
      (("b",  1.0), 0.01),
      (("c",  4.0), 100.0),
      (("d",  5.0), 100.0)
    ).map({ case ((s, y), w) => Weighted(Example(s, y), w) })
    val feats = Set("a", "b", "c", "d")
    val model = RegressionTree(OneHot(feats), lambda0, maxDepth = 1)
    val splits = model.categoricalSplitsAndErrors(data, feats)
    val (bestSplit, _) = splits.minBy(_._2._1)
    val expected = Set("a", "b", "c")
    bestSplit.features.toSet should be (expected)
  }

  it should "serialize and deserialize to/from json" in {
    val tree = regressionModel.fit(data)
    val serialized = tree.asJson.noSpaces
    implicit val fs = abFeats
    val deserialized = DecisionTree.fromJson[MixedMap[String], Double](serialized)
    deserialized should equal (tree)
  }
}
