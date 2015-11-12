package com.samthomson

import spire.algebra.{AdditiveMonoid, Field}
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
  case class Stats[+N](weight: Double, mean: N)

  object Stats {
    def of[N: Field](x: Weighted[N]): Stats[N] = Stats(x.weight, x.unweighted)
    def of[N: Field](xs: TraversableOnce[Weighted[N]]): Stats[N] = {
      val am = hasAdditiveMonoid[N]
      xs.toIterator.map(wx => of(wx)).foldLeft(am.zero)(am.plus)
    }

    def hasAdditiveMonoid[N](implicit f: Field[N]): AdditiveMonoid[Stats[N]] = new AdditiveMonoid[Stats[N]] {
      override def zero: Stats[N] = Stats(0.0, f.zero)
      override def plus(x: Stats[N], y: Stats[N]): Stats[N] = {
        // numerically stable way of computing
        // (a.weight * a.mean + b.weight * b.mean) / x.weight + y.weight
        val newWeight = x.weight + y.weight
        if (newWeight == 0.0) zero else Stats(newWeight, x.mean + (y.mean - x.mean) * (y.weight / newWeight))
      }
    }
  }
}


object WeightedMse {
  def of[N: Field](xs: TraversableOnce[Weighted[N]]): N = Stats.of(xs).meanSquaredError

  // sufficient statistic for calculating online mean squared error
  case class Stats[+N](weight: Double, mean: N, meanSquaredError: N)

  object Stats {
    def of[N: Field](wx: Weighted[N]): Stats[N] = Stats(wx.weight, wx.unweighted, Field[N].zero)
    def of[N: Field](examples: TraversableOnce[Weighted[N]]): Stats[N] = {
      val am = hasAdditiveMonoid[N]
      examples.toIterator.map(wx => of(wx)).foldLeft(am.zero)(am.plus)
    }

    def hasAdditiveMonoid[N](implicit f: Field[N]): AdditiveMonoid[Stats[N]] = new AdditiveMonoid[Stats[N]] {
      override def zero: Stats[N] = Stats(0.0, f.zero, f.zero)
      override def plus(x: Stats[N], y: Stats[N]): Stats[N] = {
        val newWeight = x.weight + y.weight
        if (newWeight == 0.0) zero else {
          // numerically stable way of computing weighted avg of `a.mean` and `b.mean`
          val newMean = x.mean + (y.mean - x.mean) * (y.weight / newWeight)
          val newVariance = {
            val aBias = x.mean - newMean
            val bBias = y.mean - newMean
            val aError = (aBias * aBias) + x.meanSquaredError
            val bError = (bBias * bBias) + y.meanSquaredError
            // numerically stable way of computing weighted avg of `aError` and `bError`
            aError + (bError - aError) * (y.weight / newWeight)
          }
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
    val am = WeightedMean.Stats.hasAdditiveMonoid[N]
    xs.toIterator.map(wx => WeightedMean.Stats.of(wx)).scanLeft(am.zero)(am.plus).map(_.mean)
  }
}
