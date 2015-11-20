package com.samthomson.decisiontree

import scala.language.implicitConversions

trait FeatureSet[F, +V, -X] {
  def feats: Set[F]
  def get(x: X)(feat: F): V
  // derived:
//  implicit def asFunction(x: X): (F => V) = { f => get(f)(x) }
}
object FeatureSet {
  type Binary[F, -X] = FeatureSet[F, Boolean, X]
  type Continuous[F, -X] = FeatureSet[F, Double, X]

  def apply[F, V, X](fs: Set[F])(implicit g: X => (F => V)): FeatureSet[F, V, X] = new FeatureSet[F, V, X] {
    override val feats: Set[F] = fs
    override def get(x: X)(feat: F): V = g(x)(feat)
  }

  def empty[F]: FeatureSet[F, Nothing, Any] = {
    FeatureSet[F, Nothing, Any](Set())(f => _ => throw new NoSuchElementException("key not found: " + f))
  }

  def oneHot[F](xs: Set[F]): Binary[F, F] = FeatureSet(xs)(a => b => a == b)

  def concat[F1, F2, V, X, Y](implicit FX: FeatureSet[F1, V, X], FY: FeatureSet[F2, V, Y]): FeatureSet[Either[F1, F2], V, (X, Y)] = {
    new FeatureSet[Either[F1, F2], V, (X, Y)] {
      override val feats: Set[Either[F1, F2]] = FX.feats.map(Left(_)) ++ FY.feats.map(Right(_))
      override def get(xy: (X, Y))(feat: Either[F1, F2]) = (xy, feat) match {
        case ((x, _), Left(f)) => FX.get(x)(f)
        case ((_, y), Right(f)) => FY.get(y)(f)
      }
    }
  }

  trait Mixed[F, -X] {
    def binary: Binary[F, X]
    def continuous: Continuous[F, X]
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

case class MixedMap[F](binaryMap: Map[F, Boolean], continuousMap: Map[F, Double])

object MixedMap {
  def concat[F](a: MixedMap[F], b: MixedMap[F]): MixedMap[F] = {
    MixedMap(a.binaryMap ++ b.binaryMap, a.continuousMap ++ b.continuousMap)
  }
  def feats[F](binaryFeats: Set[F], continuousFeats: Set[F]): FeatureSet.Mixed[F, MixedMap[F]] = FeatureSet.Mixed(
    FeatureSet(binaryFeats)(mm => f => mm.binaryMap.getOrElse(f, false)),
    FeatureSet(continuousFeats)(mm => f => mm.continuousMap.getOrElse(f, 0.0))
  )
}
