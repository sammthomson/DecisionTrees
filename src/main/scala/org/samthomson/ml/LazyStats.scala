package org.samthomson.ml

import spire.algebra.{AdditiveAbGroup, Field}
import spire.implicits._

import scala.language.implicitConversions


case class Weighted[+T](unweighted: T, weight: Double) {
  def map[B](f: T => B): Weighted[B] = Weighted(f(unweighted), weight)
}
object Weighted {
  implicit def unwrap[T](weighted: Weighted[T]): T = weighted.unweighted
}


object WeightedMean {
  def of[N: Field](xs: TraversableOnce[Weighted[N]]): N = Stats.of(xs).mean

  // sufficient statistic for calculating online mean
  case class Stats[N](weight: Double, mean: N)(implicit F: Field[N])

  object Stats {
    def of[N: Field](x: Weighted[N]): Stats[N] = Stats(x.weight, x.unweighted)
    def of[N: Field](xs: TraversableOnce[Weighted[N]]): Stats[N] = {
      val G = hasAdditiveGroup[N]
      xs.toIterator.map(wx => of(wx)).foldLeft(G.zero)(G.plus)
    }

    implicit def hasAdditiveGroup[N](implicit f: Field[N]): AdditiveAbGroup[Stats[N]] = new AdditiveAbGroup[Stats[N]] {
      override def zero: Stats[N] = Stats(0.0, f.zero)
      override def plus(x: Stats[N], y: Stats[N]): Stats[N] = {
        val (a, b) = if (x.weight.abs > y.weight.abs) (x, y) else (y, x)  // more stable this way
        val newWeight = a.weight + b.weight
        if (a.weight == 0.0) b else if (b.weight == 0.0) a else if (newWeight == 0.0) zero else {
          // numerically stable way of computing
          // (a.weight * a.mean + b.weight * b.mean) / (a.weight + b.weight)
          Stats(newWeight, a.mean + (b.mean - a.mean) * (b.weight / newWeight))
        }
      }
      override def negate(x: Stats[N]): Stats[N] = x.copy(weight = -x.weight)
    }
  }
}


object WeightedMse {
  def of[N: Field](xs: TraversableOnce[Weighted[N]]): N = Stats.of(xs).variance

  /** Sufficient statistics for calculating (online) weighted mean squared error
    *
    * @param weight the total weight of all data summarized
    * @param mean the weighted mean of `X` (i.e. 1st moment)
    * @param variance the weighted mean of `(X - mean)^2^` (i.e. 2nd central moment)
    */
  case class Stats[N](weight: Double, mean: N, variance: N)(implicit F: Field[N]) {
    def meanSquare: N = variance + mean * mean
    def error: N = variance * weight
  }

  object Stats {
    def of[N: Field](wx: Weighted[N]): Stats[N] = Stats(wx.weight, wx.unweighted, Field[N].zero)  // wx.unweighted * wx.unweighted)
    def of[N: Field](examples: TraversableOnce[Weighted[N]]): Stats[N] = {
      val G = hasAdditiveGroup[N]
      examples.toIterator.map(wx => Stats.of(wx)).foldLeft(G.zero)(G.plus)
    }

    implicit def hasAdditiveGroup[N](implicit f: Field[N]): AdditiveAbGroup[Stats[N]] = new AdditiveAbGroup[Stats[N]] {
      override val zero: Stats[N] = Stats(0.0, f.zero, f.zero)
      override def negate(x: Stats[N]): Stats[N] = x.copy(weight = -x.weight)
      override def plus(x: Stats[N], y: Stats[N]): Stats[N] = {
        val (a, b) = if (x.weight.abs > y.weight.abs) (x, y) else (y, x)  // more stable this way
        val newWeight = a.weight + b.weight
        if (a.weight == 0.0) b else if (b.weight == 0.0) a else if (newWeight == 0.0) zero else {
          val aProportion = a.weight / newWeight
          val bProportion = b.weight / newWeight
          val meanDiff = b.mean - a.mean
          // numerically stable way of computing weighted avg of `a.mean` and `b.mean`
          val newMean = a.mean + bProportion * meanDiff
          val newVariance = a.variance + bProportion * ((b.variance + aProportion * (meanDiff * meanDiff)) - a.variance)
          Stats(newWeight, newMean, newVariance)
        }
      }
    }
  }
}


object LazyStats {
  def runningMean[N: Field](xs: TraversableOnce[N]): Iterator[N] = {
    weightedRunningMean(xs.toIterator.map(Weighted(_, 1.0)))
  }

  def mean[N: Field](xs: TraversableOnce[N]): N = {
    weightedMean(xs.toIterator.map(Weighted(_, 1.0)))
  }

  def weightedMean[N: Field](xs: TraversableOnce[Weighted[N]]): N  = WeightedMean.of(xs)

  def weightedRunningMean[N: Field](xs: TraversableOnce[Weighted[N]]): Iterator[N]  = {
    val G = WeightedMean.Stats.hasAdditiveGroup[N]
    xs.toIterator.map(wx => WeightedMean.Stats.of(wx)).scanLeft(G.zero)(G.plus).map(_.mean)
  }
}
