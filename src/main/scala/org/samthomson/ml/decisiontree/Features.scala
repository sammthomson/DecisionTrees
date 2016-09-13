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
  def feats: Set[K]
  def get(x: X)(feat: K): V
  // derived:
//  implicit def asFunction(x: X): (F => V) = { f => get(f)(x) }
}
object FeatureSet {
  type Binary[K, -X] = FeatureSet[K, Boolean, X]
  type Continuous[K, -X] = FeatureSet[K, Double, X]

  def apply[K, V, X](fs: Set[K])(implicit g: X => (K => V)): FeatureSet[K, V, X] = new FeatureSet[K, V, X] {
    override val feats: Set[K] = fs
    override def get(x: X)(feat: K): V = g(x)(feat)
  }

  def empty[K]: FeatureSet[K, Nothing, Any] = {
    FeatureSet[K, Nothing, Any](Set())(f => _ => throw new NoSuchElementException("key not found: " + f))
  }

  def concat[K1, K2, V, X, Y](implicit
                              FX: FeatureSet[K1, V, X],
                              FY: FeatureSet[K2, V, Y]): FeatureSet[Either[K1, K2], V, (X, Y)] = {
    new FeatureSet[Either[K1, K2], V, (X, Y)] {
      override val feats: Set[Either[K1, K2]] = FX.feats.map(Left(_)) ++ FY.feats.map(Right(_))
      override def get(xy: (X, Y))(feat: Either[K1, K2]) = (xy, feat) match {
        case ((x, _), Left(f)) => FX.get(x)(f)
        case ((_, y), Right(f)) => FY.get(y)(f)
      }
    }
  }

  case class OneHot[K](xs: Set[K]) extends Binary[K, K] {
    override def feats: Set[K] = xs
    override def get(x: K)(feat: K): Boolean = feat == x
  }

  trait Mixed[K, -X] {
    def binary: Binary[K, X]
    def continuous: Continuous[K, X]
  }

  object Mixed {
    def apply[F, X](b: Binary[F, X], c: Continuous[F, X]): Mixed[F, X] = new Mixed[F, X] {
      override val binary = b
      override val continuous = c
    }
    implicit def fromBinary[F, X](b: Binary[F, X]): Mixed[F, X] = Mixed(b, empty)
    implicit def fromContinuous[F, X](c: Continuous[F, X]): Mixed[F, X] = Mixed(empty, c)

    def concat[F1, F2, X, Y](implicit FX: Mixed[F1, X], FY: Mixed[F2, Y]): Mixed[Either[F1, F2], (X, Y)] = {
      new Mixed[Either[F1, F2], (X, Y)] {
        override def binary = FeatureSet.concat(FX.binary, FY.binary)
        override def continuous = FeatureSet.concat(FX.continuous, FY.continuous)
      }
    }
  }
}

@SerialVersionUID(1L)
case class MixedMap[F](binarySet: Set[F], continuousMap: Map[F, Double])

object MixedMap {
  def concat[F](a: MixedMap[F], b: MixedMap[F]): MixedMap[F] = {
    MixedMap(a.binarySet ++ b.binarySet, a.continuousMap ++ b.continuousMap)
  }
  def feats[F](binaryFeats: Set[F], continuousFeats: Set[F]): FeatureSet.Mixed[F, MixedMap[F]] = FeatureSet.Mixed(
    FeatureSet(binaryFeats)(mm => f => mm.binarySet.contains(f)),
    FeatureSet(continuousFeats)(mm => f => mm.continuousMap.getOrElse(f, 0.0))
  )
}
