package com.samthomson.ml

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
  case class Stats[N] private (weight: Double, mean: N)

  object Stats {
    def of[N: Field](x: Weighted[N]): Stats[N] = Stats(x.weight, x.unweighted)
    def of[N: Field](xs: TraversableOnce[Weighted[N]]): Stats[N] = {
      val am = hasAdditiveMonoid[N]
      xs.toIterator.map(wx => of(wx)).foldLeft(am.zero)(am.plus)
    }

    def hasAdditiveMonoid[N](implicit f: Field[N]): AdditiveMonoid[Stats[N]] = new AdditiveMonoid[Stats[N]] {
      override def zero: Stats[N] = Stats(0.0, f.zero)
      override def plus(a: Stats[N], b: Stats[N]): Stats[N] = {
        if (a.weight == 0.0) b else if (b.weight == 0.0) a else {
          val newWeight = a.weight + b.weight
          // numerically stable way of computing
          // (a.weight * a.mean + b.weight * b.mean) / (a.weight + b.weight)
          Stats(newWeight, a.mean + (b.mean - a.mean) * (b.weight / newWeight))
        }
      }
    }
  }
}


object WeightedMse {
  def of[N: Field](xs: TraversableOnce[Weighted[N]]): N = Stats.of(xs).meanSquaredError

  // sufficient statistic for calculating online mean squared error
  case class Stats[N] private (weight: Double, mean: N, meanSquaredError: N) {
    def error(implicit F: Field[N]): N = meanSquaredError * weight
  }

  object Stats {
    def of[N: Field](wx: Weighted[N]): Stats[N] = Stats(wx.weight, wx.unweighted, Field[N].zero)
    def of[N: Field](examples: TraversableOnce[Weighted[N]]): Stats[N] = {
      val am = hasAdditiveMonoid[N]
      examples.toIterator.map(wx => Stats.of(wx)).foldLeft(am.zero)(am.plus)
    }

    def hasAdditiveMonoid[N](implicit f: Field[N]): AdditiveMonoid[Stats[N]] = new AdditiveMonoid[Stats[N]] {
      override val zero: Stats[N] = Stats(0.0, f.zero, f.zero)
      override def plus(a: Stats[N], b: Stats[N]): Stats[N] = {
        val newWeight = a.weight + b.weight
        if (a.weight == 0.0) b else if (b.weight == 0.0) a else {
          val bProportion = b.weight / newWeight
          // numerically stable way of computing weighted avg of `a.mean` and `b.mean`
          val newMean = a.mean + (b.mean - a.mean) * bProportion
          val newMse = {
            val aBias = a.mean - newMean
            val bBias = b.mean - newMean
            val aError = (aBias * aBias) + a.meanSquaredError
            val bError = (bBias * bBias) + b.meanSquaredError
            // numerically stable way of computing weighted avg of `aError` and `bError`
            aError + (bError - aError) * bProportion
          }
          Stats(newWeight, newMean, newMse)
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
