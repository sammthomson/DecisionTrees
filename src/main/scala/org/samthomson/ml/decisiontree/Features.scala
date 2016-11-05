package org.samthomson.ml.decisiontree

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

  def apply[K, V, X](g: X => Map[K, V]): FeatureSet[K, V, X] = new FeatureSet[K, V, X] {
    override def featVals(x: X): Map[K, V] = g(x)
    override def get(feat: K)(x: X): V = g(x)(feat)
  }

  def empty[K]: FeatureSet[K, Nothing, Any] = {
    FeatureSet[K, Nothing, Any](_ => Map.empty)
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

  def sparse[X, K](f: X => Set[K]): Binary[K, X] = new Binary[K, X] {
    override def get(feat: K)(x: X): Boolean = f(x).contains(feat)
    override def featVals(x: X): Map[K, Boolean] = f(x).iterator.map(_ -> true).toMap.withDefaultValue(false)
  }

  def oneHot[K](feats: Set[K]): Binary[K, K] = sparse(Set(_))

  case class Composed[A, B, K, V](featureSet: FeatureSet[K, V, B], f: A => B) extends FeatureSet[K, V, A] {
    override def get(feat: K)(x: A): V = featureSet.get(feat)(f(x))
    override def featVals(x: A): Map[K, V] = featureSet.featVals(f(x))
  }

  case class Mixed[K, -X](binary: Binary[K, X],
                          continuous: Continuous[K, X],
                          categorical: Categorical[K, X]) {
    def compose[B](f: B => X): Mixed[K, B] = Mixed(
      binary compose f,
      continuous compose f,
      categorical compose f
    )
  }

  object Mixed {
    implicit def fromBinary[F, X](b: Binary[F, X]): Mixed[F, X] = Mixed(b, empty, empty)
    implicit def fromContinuous[F, X](c: Continuous[F, X]): Mixed[F, X] = Mixed(empty, c, empty)
    implicit def fromCategorical[F, X](c: Categorical[F, X]): Mixed[F, X] = Mixed(empty, empty, c)

    def concat[F1, F2, X, Y](implicit FX: Mixed[F1, X], FY: Mixed[F2, Y]): Mixed[Either[F1, F2], (X, Y)] = {
      Mixed[Either[F1, F2], (X, Y)](
        FeatureSet.concat(FX.binary, FY.binary),
        FeatureSet.concat(FX.continuous, FY.continuous),
        FeatureSet.concat(FX.categorical, FY.categorical)
      )
    }
  }
}

@SerialVersionUID(1L)
case class MixedMap[F](binarySet: Set[F],
                       continuousMap: Map[F, Double],
                       categoricalMap: Map[F, String]) {
  def filter(p: F => Boolean): MixedMap[F] = MixedMap(
    binarySet.filter(p),
    continuousMap.filterKeys(p),
    categoricalMap.filterKeys(p)
  )
  def map[B](f: F => B): MixedMap[B] = MixedMap(
    binarySet.map(f),
    continuousMap.map { case (k, v) => (f(k), v) },
    categoricalMap.map { case (k, v) => (f(k), v) }
  )
  def ++(other: MixedMap[F]): MixedMap[F] = MixedMap(
    binarySet ++ other.binarySet,
    continuousMap ++ other.continuousMap,
    categoricalMap ++ other.categoricalMap
  )
}

object MixedMap {
  implicit def featSet[F]: FeatureSet.Mixed[F, MixedMap[F]] = {
    FeatureSet.Mixed(
      FeatureSet.sparse(_.binarySet),
      FeatureSet(_.continuousMap.withDefaultValue(0.0)),
      FeatureSet(_.categoricalMap)
    )
  }

  def empty[F]: MixedMap[F] = MixedMap(Set(), Map(), Map())

  def binary[F](ks: Set[F]): MixedMap[F] = MixedMap(ks, Map(), Map())
  def continuous[F](ks: Map[F, Double]): MixedMap[F] = MixedMap(Set(), ks, Map())
  def categorical[F](ks: Map[F, String]): MixedMap[F] = MixedMap(Set(), Map(), ks)
}
