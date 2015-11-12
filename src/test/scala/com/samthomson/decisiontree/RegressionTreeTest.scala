package com.samthomson.decisiontree

import com.samthomson.Weighted
import com.samthomson.decisiontree.RegressionTree.fit
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, FlatSpec}

object RegressionTreeTest {
  val features = Set("a", "b")
  val data = Vector(
    Example(Map("a" -> -1.0, "b" -> 1.0), -1.0),
    Example(Map("a" ->  0.0, "b" -> 1.0),  1.0),
    Example(Map("a" ->  1.0, "b" -> 1.0),  5.0),
    Example(Map("a" ->  1.1, "b" -> 1.0),  4.0)
  ).map(Weighted(_, 1.0))
}
class RegressionTreeTest extends FlatSpec with Matchers with GeneratorDrivenPropertyChecks {
  import RegressionTreeTest._

  "RegressionTree.fitRegression" should "fit perfectly given enough depth" in {
    val tree = fit(data, features, 3)
    tree.depth should be (3)
    for (d <- data) {
      tree.predict(d.input) should be (d.output)
    }
  }

  it should "respect maxDepth" in {
    forAll { (rawData: Vector[(Double, Double, Double, Double)]) =>
      val data = rawData.map({ case (a, b, o, w) => Weighted(Example(Map("a" -> a, "b" -> b), o), math.abs(w)) })
      val tree = fit(data, features, 2)
      tree.depth should be <= 2
    }
  }

  it should "fit approximately given a little depth" in {
    val tree = fit(data, features, 2)
    for (d <- data) {
      tree.predict(d.input) should be (d.output +- 1.0)
    }
  }

  it should "stop when there is 0 error" in {
    val constantData = data.map(_.map(_.copy(output = 2.3)))
    val tree = fit(constantData, features, 5)
    tree.depth should be (1)
    for (d <- constantData) {
      tree.predict(d.input) should be (d.output)
    }
  }
}
