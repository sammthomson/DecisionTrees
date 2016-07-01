package org.samthomson.ml.decisiontree

import io.circe._
import io.circe.syntax._
import org.samthomson.ml.LazyStats.weightedMean
import org.samthomson.ml.Weighted
import spire.algebra.Field
import scala.math.max


@SerialVersionUID(1L)
case class Example[+X, +Y](input: X, output: Y)


@SerialVersionUID(1L)
sealed trait Splitter[-X] extends Function[X, Boolean] {

  def isLeft(input: X): Boolean
  // derived:
  def choose[T](input: X)(left: T, right: T): T = if (isLeft(input)) left else right
  def apply(input: X): Boolean = isLeft(input)
}
object Splitter {
  // JSON codec
  implicit def encoder[X]: Encoder[Splitter[X]] = Encoder.instance {
      // TODO: feature type `F` isn't visible, so we just call toString on feats.
      // works for most basic feature types, but it's pretty hacky.
      case FeatureThreshold(f, t) => Json.obj(
        "feature" -> Json.fromString(f.toString),
        "threshold" -> Json.fromDoubleOrString(t)
      )
      case BoolSplitter(f) => Json.obj("feature" -> Json.fromString(f.toString))
      case OrSplitter(fs) => Json.obj("features" -> fs.map(_.toString).asJson)
    }
  implicit def decoder[F, X](implicit feats: FeatureSet.Mixed[F, X],
                             fDec: Decoder[F]): Decoder[Splitter[X]] = {
    val threshDec: Decoder[Splitter[X]] = Decoder.instance { cursor =>
      for (
        feature <- cursor.get[F]("feature");
        threshold <- cursor.get[Double]("threshold")
      ) yield FeatureThreshold[F, X](feature, threshold)(feats.continuous)
    }
    val orDec: Decoder[Splitter[X]] = Decoder.instance(_.get[Iterable[F]]("features").map(OrSplitter(_)(feats.binary)))
    val boolDec: Decoder[Splitter[X]] = Decoder.instance(_.get[F]("feature").map(BoolSplitter(_)(feats.binary)))
    threshDec or orDec or boolDec
  }
}
case class FeatureThreshold[+F, X](feature: F,
                                   threshold: Double)
                                  (implicit cf: FeatureSet.Continuous[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = cf.get(input)(feature) <= threshold
  override def toString: String = s"$feature <= $threshold"
}
case class BoolSplitter[+F, -X](feature: F)
                               (implicit bf: FeatureSet.Binary[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = bf.get(input)(feature)
  override def toString: String = s"$feature"
}
case class OrSplitter[+F, -X](features: Iterable[F])
                             (implicit bf: FeatureSet.Binary[F, X]) extends Splitter[X] {
  override def isLeft(input: X): Boolean = features.exists(bf.get(input))
  override def toString: String = s"or(${features.mkString(", ")})"
}


@SerialVersionUID(1L)
sealed trait DecisionTree[-X, +Y] extends Model[X, Y] {
  def depth: Int
  def numNodes: Int
  def predict(input: X): Y
  def prettyPrint(indent: String = ""): String
  override def toString: String = prettyPrint()
}
object DecisionTree {
  def fromJson[X, Y](json: String)
                    (implicit yDec: Decoder[Y],
                     sDec: Decoder[Splitter[X]]): DecisionTree[X, Y] =
    jawn.decode[DecisionTree[X, Y]](json).valueOr(throw _)
  // JSON codec
  implicit def encoder[X, Y: Encoder]: Encoder[DecisionTree[X, Y]] = Encoder.instance {
      case Leaf(y) => y.asJson
      case Split(splitter, left, right) => Json.obj(
        "split" -> splitter.asJson,
        "left" -> left.asJson,
        "right" -> right.asJson
      )
    }
  implicit def decoder[X, Y](implicit
                             yDec: Decoder[Y],
                             sDec: Decoder[Splitter[X]]): Decoder[DecisionTree[X, Y]] = {
    val leafDecoder: Decoder[DecisionTree[X, Y]] = Decoder.instance(_.as[Y].map(Leaf(_)))
    val splitDecoder: Decoder[DecisionTree[X, Y]] = Decoder.instance { cursor =>
      for (
        splitter <- cursor.get[Splitter[X]]("split");
        left <- cursor.get[DecisionTree[X, Y]]("left");
        right <- cursor.get[DecisionTree[X, Y]]("right")
      ) yield
        Split(splitter, left, right)
    }
    leafDecoder or splitDecoder
  }
}
case class Split[-X, Y](split: Splitter[X],
                        left: DecisionTree[X, Y],
                        right: DecisionTree[X, Y]) extends DecisionTree[X, Y] {
  override lazy val depth = max(left.depth, right.depth) + 1
  override lazy val numNodes = left.numNodes + right.numNodes + 1
  final override def predict(input: X): Y = split.choose(input)(left, right).predict(input)
  override def prettyPrint(indent: String): String = {
    val indented = indent + "  "
    indent + s"Split(\n" +
        indented + split + "\n" +
        left.prettyPrint(indented) + ",\n" +
        right.prettyPrint(indented) + "\n" +
    indent + ")"
  }
}
case class Leaf[Y](constant: Y) extends DecisionTree[Any, Y] {
  override val depth = 1
  override val numNodes = 1
  final override def predict(input: Any): Y = constant
  override def prettyPrint(indent: String): String = indent + s"Leaf($constant)"
}
object Leaf {
  def averaging[X, Y: Field](examples: Iterable[Weighted[Example[X, Y]]]): Leaf[Y] = {
    val weightedOutputs = examples.map(_.map(_.output))
    Leaf(weightedMean(weightedOutputs))
  }
}
