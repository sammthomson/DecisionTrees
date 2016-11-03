package org.samthomson.ml.decisiontree

import org.samthomson.ml.decisiontree.FeatureSet.SparseBinary

import scala.language.implicitConversions


/**
  * Evidence that `F`s hold features of type `K` with values of type `V`.
 *
  * @tparam K the type of features
  * @tparam V the type of feature values
  * @tparam X the type of objects that map `K => V`
  */
@SerialVersionUID(1L)
trait FeatureSet[K, +V, -X] extends Serializable {
  def get(feat: K)(x: X): V
  def featVals(x: X): Map[K, V]

  def compose[B](f: B => X): FeatureSet[K, V, B] = FeatureSet.Composed(this, f)
}
object FeatureSet {
  type Binary[K, -X] = FeatureSet[K, Boolean, X]
  type Continuous[K, -X] = FeatureSet[K, Double, X]
  type Categorical[K, -X] = FeatureSet[K, String, X]  // TODO: r/String/???

  def apply[K, V, X](fs: Set[K])
                    (g: X => Map[K, V]): FeatureSet[K, V, X] = new FeatureSet[K, V, X] {
    override def featVals(x: X): Map[K, V] = g(x)
    override def get(feat: K)(x: X): V = g(x)(feat)
  }

  def empty[K]: FeatureSet[K, Nothing, Any] = {
    FeatureSet[K, Nothing, Any](Set())(f => Map.empty)
  }

  def concat[K1, K2, V, X, Y](implicit
                              FX: FeatureSet[K1, V, X],
                              FY: FeatureSet[K2, V, Y]): FeatureSet[Either[K1, K2], V, (X, Y)] = {
    new FeatureSet[Either[K1, K2], V, (X, Y)] {
      override def get(feat: Either[K1, K2])(xy: (X, Y)) = (xy, feat) match {
        case ((x, _), Left(f)) => FX.get(f)(x)
        case ((_, y), Right(f)) => FY.get(f)(y)
      }
      override def featVals(xy: (X, Y)): Map[Either[K1, K2], V] = {
        val (x, y) = xy
        Map() ++
            FX.featVals(x).map { case (k, v) => Left(k)  -> v } ++
            FY.featVals(y).map { case (k, v) => Right(k) -> v }
      }
    }
  }

  case class SparseBinary[K](feats: Set[K]) extends Binary[K, Set[K]] {
    override def get(feat: K)(x: Set[K]): Boolean = x.contains(feat)
    override def featVals(x: Set[K]): Map[K, Boolean] = x.iterator.map(_ -> true).toMap.withDefaultValue(false)
  }

  def oneHot[K](feats: Set[K]): Binary[K, K] = SparseBinary(feats) compose (Set(_))

  case class Composed[A, B, K, V](featureSet: FeatureSet[K, V, B], f: A => B) extends FeatureSet[K, V, A] {
    override def get(feat: K)(x: A): V = featureSet.get(feat)(f(x))
    override def featVals(x: A): Map[K, V] = featureSet.featVals(f(x))
  }

  trait Mixed[K, -X] {
    def binary: Binary[K, X]
    def continuous: Continuous[K, X]
    def categorical: Categorical[K, X]
  }

  object Mixed {
    def apply[F, X](b: Binary[F, X],
                    c: Continuous[F, X],
                    x: Categorical[F, X]): Mixed[F, X] = new Mixed[F, X] {
      override val binary = b
      override val continuous = c
      override val categorical = x
    }
    implicit def fromBinary[F, X](b: Binary[F, X]): Mixed[F, X] = Mixed(b, empty, empty)
    implicit def fromContinuous[F, X](c: Continuous[F, X]): Mixed[F, X] = Mixed(empty, c, empty)
    implicit def fromCategorical[F, X](c: Categorical[F, X]): Mixed[F, X] = Mixed(empty, empty, c)

    def concat[F1, F2, X, Y](implicit FX: Mixed[F1, X], FY: Mixed[F2, Y]): Mixed[Either[F1, F2], (X, Y)] = {
      new Mixed[Either[F1, F2], (X, Y)] {
        override def binary = FeatureSet.concat(FX.binary, FY.binary)
        override def continuous = FeatureSet.concat(FX.continuous, FY.continuous)
        override def categorical = FeatureSet.concat(FX.categorical, FY.categorical)
      }
    }
  }
}

@SerialVersionUID(1L)
case class MixedMap[F](binarySet: Set[F],
                       continuousMap: Map[F, Double],
                       categoricalMap: Map[F, String])

object MixedMap {
  def concat[F](a: MixedMap[F],
                b: MixedMap[F]): MixedMap[F] = {
    MixedMap(
      a.binarySet ++ b.binarySet,
      a.continuousMap ++ b.continuousMap,
      a.categoricalMap ++ b.categoricalMap
    )
  }
  def featSet[F](binaryFeats: Set[F],
                 continuousFeats: Set[F],
                 categoricalFeats: Set[F]): FeatureSet.Mixed[F, MixedMap[F]] = {
    FeatureSet.Mixed(
      SparseBinary(binaryFeats) compose (_.binarySet),
      FeatureSet(continuousFeats)(_.continuousMap.withDefaultValue(0.0)),
      FeatureSet(categoricalFeats)(_.categoricalMap)
    )
  }
}
