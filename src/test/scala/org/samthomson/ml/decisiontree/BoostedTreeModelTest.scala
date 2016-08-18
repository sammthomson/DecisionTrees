package org.samthomson.ml.decisiontree

import java.io.{ByteArrayInputStream, ObjectInputStream, ObjectOutputStream, ByteArrayOutputStream}

import org.samthomson.TestHelpers
import org.samthomson.ml.decisiontree.FeatureSet.Mixed._
import org.samthomson.ml.decisiontree.TwiceDiffableLoss.{MultiClassHinge, MultiClassLogLoss, MultiClassSquaredHinge}
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
    Example(MixedMap(Set("is_animal"), Map("tail_length" -> -1.0)), "dog"),
    Example(MixedMap(Set("is_animal"), Map("tail_length" ->  0.0)), "cat"),
    Example(MixedMap(Set("is_animal"), Map("tail_length" ->  1.0)), "dog"),
    Example(MixedMap(Set("is_animal"), Map("tail_length" ->  1.1)), "dog"),
    Example(MixedMap(Set("is_animal"), Map("tail_length" ->  1.2)), "cat")
  )
}

class BoostedTreeModelTest extends FlatSpec with TestHelpers with Matchers {
  import BoostedTreeModelTest._

  "GradientBoostedTreesTest.fit" should "fit perfectly using MultiClassLogLoss" in {
    val model: BoostedTreeModel[Either[String, String], MixedMap[String], String] =
      BoostedTreeModel(_ => outputSpace, xyFeats, MultiClassLogLoss(), lambda0, lambda2, 4)
    val (boostedForest, _) = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassHinge" in {
    val model: BoostedTreeModel[Either[String, String], MixedMap[String], String] =
      BoostedTreeModel(_ => outputSpace, xyFeats, MultiClassHinge(), lambda0, lambda2, 4)
    val (boostedForest, _) = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "fit perfectly using MultiClassSquaredHinge" in {
    val model: BoostedTreeModel[Either[String, String], MixedMap[String], String] =
      BoostedTreeModel(_ => outputSpace, xyFeats, MultiClassSquaredHinge(), lambda0, lambda2, 4)
    val (boostedForest, _) = model.fit(data, 200)
    for (d <- data) {
      boostedForest.predict(d.input) should be (d.output)
    }
  }

  it should "serialize and deserialize to/from object stream" in {
    val model: BoostedTreeModel[Either[String, String], MixedMap[String], String] =
      BoostedTreeModel(_ => outputSpace, xyFeats, MultiClassSquaredHinge(), lambda0, lambda2, 4)
    val (boostedForest, _) = model.fit(data, 200)
    val serialized = new ByteArrayOutputStream()
    new ObjectOutputStream(serialized).writeObject(boostedForest)
    val deserialized = {
      new ObjectInputStream(new ByteArrayInputStream(serialized.toByteArray))
          .readObject().asInstanceOf[MultiClassModel[MixedMap[String], String]]
    }
    boostedForest should equal (deserialized)
    for (d <- data) {
      boostedForest.predict(d.input) should equal (deserialized.predict(d.input))
    }
  }
}
