package com.samthomson.decisiontree

import com.samthomson.{Weighted, TestHelpers}
import org.scalatest.{Matchers, FlatSpec}
import spire.implicits._


object GradientBoostedTreesTest {
  val features = Set("a", "b")
  val outputSpace = Set(true, false)
  val data = Vector(
    Example(Map("a" -> -1.0, "b" -> 1.0), false),
    Example(Map("a" ->  0.0, "b" -> 1.0),  true),
    Example(Map("a" ->  1.0, "b" -> 1.0),  false),
    Example(Map("a" ->  1.1, "b" -> 1.0),  false),
    Example(Map("a" ->  1.2, "b" -> 1.0),  true)
  ).map(Weighted(_, 1.0))
}

class GradientBoostedTreesTest extends FlatSpec with TestHelpers with Matchers {
  import GradientBoostedTreesTest._

  "GradientBoostedTreesTest.hingeBoost" should "fit perfectly given enough depth and iterations" in {
    val boostedForest = GradientBoostedTrees.hingeBoost(data, features, outputSpace, 4, 200)
    for (d <- data) {
//      println(s"predicted: ${boostedForest.predict(d.input)} \t gold: ${d.output} \t scores: ${boostedForest.scores(d.input)}")
      boostedForest.predict(d.input) should be (d.output)
    }
  }
}
