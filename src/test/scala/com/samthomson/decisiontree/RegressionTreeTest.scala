package com.samthomson.decisiontree

import com.samthomson.Weighted
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, FlatSpec}

object RegressionTreeTest {
  val continuousFeats = Set("a", "b")
  val abFeats = MixedMap.feats(Set[String](), continuousFeats)
  private val toExample: (((Double, Double), Double)) => Example[MixedMap[String], Double] = {
    case ((a, b), y) => Example(MixedMap(Map(), Map("a" -> a, "b" -> b)), y)
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
}
