package com.samthomson

import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, FlatSpec}
import spire.implicits._



class LazyStatsTest extends FlatSpec with TestHelpers with Matchers with GeneratorDrivenPropertyChecks {

  "LazyMean.mean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      val mean = LazyStats.mean(input.iterator)
      mean should be (roughlyEqualTo(input.sum / input.size)(tolerance))
    }
  }

  "LazyMean.runningMean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      val mean = LazyStats.runningMean(input.iterator)
      for ((m, i) <- mean.zipWithIndex) {
        m should be (roughlyEqualTo(input.take(i).sum / math.max(i, 1))(tolerance))
      }
    }
  }
}
