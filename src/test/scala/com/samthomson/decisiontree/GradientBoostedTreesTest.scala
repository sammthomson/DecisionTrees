package com.samthomson.decisiontree

import com.samthomson.decisiontree.TwiceDiffableLoss.{MultiClassLogLoss, MultiClassHinge, MultiClassSquaredHinge}
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
  val lambda0 = 0.0 // 1e-7
  val lambda2 = 1.0
  val data: Vector[Weighted[Example[MixedMap[String], String]]] = Vector(
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" -> -1.0)), "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  0.0)),  "cat"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.0)),  "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.1)),  "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.2)),  "cat")
  ).map(Weighted(_, 1.0))
}

class GradientBoostedTreesTest extends FlatSpec with TestHelpers with Matchers {
  import GradientBoostedTreesTest._

  "GradientBoostedTreesTest.fit" should "fit perfectly using MultiClassLogLoss" in {
    val model = GradientBoostedTrees(outputSpace, xyFeats, MultiClassLogLoss(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassHinge" in {
    val model = GradientBoostedTrees(outputSpace, xyFeats, MultiClassHinge(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassSquaredHinge" in {
    val model = GradientBoostedTrees(outputSpace, xyFeats, MultiClassSquaredHinge(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }
}
