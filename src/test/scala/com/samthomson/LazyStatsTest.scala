package com.samthomson

import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, FlatSpec}
import spire.implicits._



class LazyStatsTest extends FlatSpec with TestHelpers with Matchers with GeneratorDrivenPropertyChecks {

  "LazyStats.mean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      val mean = LazyStats.mean(input)
      val expectedMean = input.sum / input.size
      mean should be (roughlyEqualTo(expectedMean)(tolerance))
    }
  }

  "LazyStats.runningMean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      val mean = LazyStats.runningMean(input)
      for ((m, i) <- mean.zipWithIndex) {
        m should be (roughlyEqualTo(input.take(i).sum / math.max(i, 1))(tolerance))
      }
    }
  }

  "WeightedSquaredError.of" should "be weightedMean of squaredErrors" in {
    forAll { (head: (Double, Double), tail: List[(Double, Double)]) =>
      val input = (head :: tail).map({ case (x, w) => Weighted(x, math.abs(w)) })
      val mse = WeightedMse.of(input)
      val expectedMse = {  // calculate MSE the obvious way
        val mean = LazyStats.weightedMean(input)
        val errors = input.map(_.map(x => { val diff = x - mean; diff * diff }))
        LazyStats.weightedMean(errors)
      }
      if (mse.isNaN) {  // should be at least as numerically stable as the obvious way
        expectedMse.isNaN should be (true)
      } else {
        mse should be (roughlyEqualTo(expectedMse)(tolerance))
      }
    }
  }
}
