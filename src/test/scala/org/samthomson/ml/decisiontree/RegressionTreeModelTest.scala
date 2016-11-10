package org.samthomson.ml.decisiontree

import io.circe.syntax._
import org.samthomson.ml.{WeightedMse, Weighted}
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{FlatSpec, Matchers}
import spire.implicits._

object RegressionTreeModelTest {
  val lambda0 = 0.1

  val stringFeats = MixedMap.featureSet[String]
  val toExample: (((Double, Double), Double)) => Example[MixedMap[String], Double] = {
    case ((a, b), y) => Example(MixedMap.continuous(Map("a" -> a, "b" -> b)), y)
  }
  val data = Vector(
    ((-1.0, 1.0), -1.0),
    (( 0.0, 1.0),  1.0),
    (( 1.0, 1.0),  5.0),
    (( 1.1, 1.0),  4.0)
  ).map(toExample).map(Weighted(_, 1.0))
  val toCategoricalExample: ((String, Double)) => Example[MixedMap[String], Double] = {
    case (v, y) => Example(MixedMap.categorical(Map("X" -> v)), y)
  }
  val categoricalData = Vector(
    ("a", -1.0),
    ("b", 1.0),
    ("c", 4.0),
    ("d", 5.0)
  ).map(toCategoricalExample)
  val evenCategoricalData = categoricalData.map(Weighted(_, 1.0))
  val unevenCategoricalData = categoricalData.
      zip(Vector(0.01, 0.01, 100.0, 100.0)).
      map { case (e, w) => Weighted(e, w) }
}

class RegressionTreeModelTest extends FlatSpec with Matchers with GeneratorDrivenPropertyChecks {
  import RegressionTreeModelTest._
  val regressionModel = RegressionTreeModel(stringFeats, lambda0, maxDepth = 3)

  "RegressionTree.fitRegression" should "fit perfectly given enough depth" in {
    val (tree, loss) = regressionModel.fit(data)
    tree.depth should equal (3)
    for (d <- data) {
      tree.predict(d.input) should equal (d.output)
    }
    loss should be <= 0.0
  }

  it should "respect maxDepth" in {
    forAll { (rawData: Vector[(((Double, Double), Double), Double)]) =>
      val data = rawData.map({ case (e, w) => Weighted(toExample(e), math.abs(w)) })
      val (tree, _) = regressionModel.copy(maxDepth = 2).fit(data)
      tree.depth should be <= 2
    }
  }

  it should "fit approximately given a little depth" in {
    val (tree, loss) = regressionModel.copy(maxDepth = 2).fit(data)
    for (d <- data) {
      tree.predict(d.input) should be (d.output +- 1.0)
    }
    loss should be <= 2.5
  }

  it should "stop when there is 0 error" in {
    val constantData = data.map(_.map(_.copy(output = 2.3)))
    val (tree, loss) = regressionModel.copy(maxDepth = 5).fit(constantData)
    tree.depth should be (1)
    for (d <- constantData) {
      tree.predict(d.input) should be (d.output)
    }
    loss should be <= 0.0
  }

  "categoricalSplitsAndErrors" should "find subsets of features with equal weights" in {
    val db = IndexedExamples.build(evenCategoricalData)(stringFeats)
    val model = RegressionTreeModel(stringFeats, lambda0, maxDepth = 1)
    val splits = model.categoricalSplitsAndErrors(
      db,
      db.inputs.indices.toSet,
      WeightedMse.Stats.of(db.outputs)
    )
    val (bestSplit, _) = splits.minBy(_._2.totalErrAndEvenness.error)
    val expected = Set("a", "b")
    bestSplit.values should be (expected)
  }

  it should "find subsets of features with unequal weights" in {
    val db = IndexedExamples.build(unevenCategoricalData)(stringFeats)
    val model = RegressionTreeModel(stringFeats, lambda0, maxDepth = 1)
    val splits = model.categoricalSplitsAndErrors(
      db,
      db.inputs.indices.toSet,
      WeightedMse.Stats.of(db.outputs)
    )
    val (bestSplit, _) = splits.minBy(_._2.totalErrAndEvenness.error)
    val expected = Set("a", "b", "c")
    bestSplit.values should be (expected)
  }

  "RegressionTree" should "serialize and deserialize to/from json" in {
    implicit val fs = stringFeats
    val model = RegressionTreeModel(stringFeats, lambda0, maxDepth = 4)
    val (tree, _) = model.fit(evenCategoricalData)
    val serialized = tree.asJson.noSpaces
    val deserialized = DecisionTree.fromJson[MixedMap[String], Double](serialized)
    deserialized should equal (tree)
  }
}
