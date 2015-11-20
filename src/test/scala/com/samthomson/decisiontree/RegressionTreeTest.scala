package com.samthomson.decisiontree

import com.samthomson.Weighted
import com.samthomson.decisiontree.RegressionTree.fit
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, FlatSpec}

object RegressionTreeTest {
  val continuousFeats = Set("a", "b")
  implicit val abFeats = MixedMap.feats(Set[String](), continuousFeats)
  private val toExample: (((Double, Double), Double)) => Example[MixedMap[String], Double] = {
    case ((a, b), y) => Example(MixedMap(Map(), Map("a" -> a, "b" -> b)), y)
  }
  val data = Vector(
    ((-1.0, 1.0), -1.0),
    (( 0.0, 1.0),  1.0),
    (( 1.0, 1.0),  5.0),
    (( 1.1, 1.0),  4.0)
  ).map(toExample).map(Weighted(_, 1.0))
}
class RegressionTreeTest extends FlatSpec with Matchers with GeneratorDrivenPropertyChecks {
  import RegressionTreeTest._

  "RegressionTree.fitRegression" should "fit perfectly given enough depth" in {
    val tree = fit(data, 3)
    tree.depth should be (3)
    for (d <- data) {
      tree.predict(d.input) should be (d.output)
    }
  }

  it should "respect maxDepth" in {
    forAll { (rawData: Vector[(((Double, Double), Double), Double)]) =>
      val data = rawData.map({ case (e, w) => Weighted(toExample(e), math.abs(w)) })
      val tree = fit(data, 2)
      tree.depth should be <= 2
    }
  }

  it should "fit approximately given a little depth" in {
    val tree = fit(data, 2)
    for (d <- data) {
      tree.predict(d.input) should be (d.output +- 1.0)
    }
  }

  it should "stop when there is 0 error" in {
    val constantData = data.map(_.map(_.copy(output = 2.3)))
    val tree = fit(constantData, 5)
    tree.depth should be (1)
    for (d <- constantData) {
      tree.predict(d.input) should be (d.output)
    }
  }
}
