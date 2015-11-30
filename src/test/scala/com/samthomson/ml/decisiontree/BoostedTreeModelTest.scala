package com.samthomson.ml.decisiontree

import com.samthomson.TestHelpers
import com.samthomson.ml.decisiontree.FeatureSet.Mixed._
import com.samthomson.ml.decisiontree.TwiceDiffableLoss.{MultiClassHinge, MultiClassLogLoss, MultiClassSquaredHinge}
import org.scalatest.{FlatSpec, Matchers}

object BoostedTreeModelTest {
  val outputSpace = Set("cat", "dog")
  val xyFeats = {
    val inputFeats = MixedMap.feats(Set("is_animal"), Set("tail_length"))
    val outputFeats = FeatureSet.OneHot(outputSpace)
    FeatureSet.Mixed.concat(inputFeats, outputFeats)
  }
  val lambda0 = 0.0 // 1e-7
  val lambda2 = 1.0
  val data: Vector[Example[MixedMap[String], String]] = Vector(
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" -> -1.0)), "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  0.0)),  "cat"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.0)),  "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.1)),  "dog"),
    Example(MixedMap(Map("is_animal" -> true), Map("tail_length" ->  1.2)),  "cat")
  )
}

class BoostedTreeModelTest extends FlatSpec with TestHelpers with Matchers {
  import BoostedTreeModelTest._

  "GradientBoostedTreesTest.fit" should "fit perfectly using MultiClassLogLoss" in {
    val model = BoostedTreeModel(outputSpace, xyFeats, MultiClassLogLoss(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassHinge" in {
    val model = BoostedTreeModel(outputSpace, xyFeats, MultiClassHinge(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassSquaredHinge" in {
    val model = BoostedTreeModel(outputSpace, xyFeats, MultiClassSquaredHinge(), lambda0, lambda2, 4)
    val boostedForest = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }
}
