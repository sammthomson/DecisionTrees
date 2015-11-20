package com.samthomson.decisiontree

import com.samthomson.{Weighted, TestHelpers}
import com.samthomson.decisiontree.FeatureSet.Mixed._
import org.scalatest.{Matchers, FlatSpec}

object GradientBoostedTreesTest {
  val outputSpace = Set("cat", "dog")
  val xyFeats = {
    val inputFeats = MixedMap.feats(Set("is_animal"), Set("tail_length"))
    val outputFeats = FeatureSet.oneHot(outputSpace)
    FeatureSet.Mixed.concat(inputFeats, outputFeats)
  }

  val data: Vector[Weighted[Example[MixedMap[String], String]]] = Vector(
    Example(MixedMap(Map() /*"is_animal" -> true)*/, Map("tail_length" -> -1.0)), "dog"),
    Example(MixedMap(Map() /*"is_animal" -> true)*/, Map("tail_length" ->  0.0)),  "cat"),
    Example(MixedMap(Map() /*"is_animal" -> true)*/, Map("tail_length" ->  1.0)),  "dog"),
    Example(MixedMap(Map() /*"is_animal" -> true)*/, Map("tail_length" ->  1.1)),  "dog"),
    Example(MixedMap(Map() /*"is_animal" -> true)*/, Map("tail_length" ->  1.2)),  "cat")
  ).map(Weighted(_, 1.0))
}

class GradientBoostedTreesTest extends FlatSpec with TestHelpers with Matchers {
  import GradientBoostedTreesTest._

  "GradientBoostedTreesTest.hingeBoost" should "fit perfectly given enough depth and iterations" in {
    val boostedForest = GradientBoostedTrees.hingeBoost(data, outputSpace, 4, 200)(xyFeats)
//    println(boostedForest)
    for (d <- data) {
      println(s"predicted: ${boostedForest.predict(d.input)} \t gold: ${d.output} \t scores: ${boostedForest.scores(d.input).toMap}")
      boostedForest.predict(d.input) should be (d.output)
    }
  }
}
