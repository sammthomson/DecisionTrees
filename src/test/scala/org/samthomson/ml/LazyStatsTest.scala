package org.samthomson.ml

import org.samthomson.TestHelpers
import WeightedMse.{Stats => MseStats}
import org.scalacheck.{Gen, Arbitrary}
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{FlatSpec, Matchers}
import spire.implicits._


object LazyStatsTest {
  implicit def arbWeighted[X : Arbitrary]: Arbitrary[Weighted[X]] = Arbitrary(
    for (
      w <- arbitrary[Float];
      x <- arbitrary[X]
    ) yield Weighted(x, w.abs)
  )
  implicit def arbMse: Arbitrary[MseStats[Double]] = Arbitrary(
    // floating point errors go wild if you let these numbers get too big
    for (
      w <- Gen.choose(-100.0, 100.0);
      mean <- Gen.choose(-100.0, 100.0);
      mse <- Gen.choose(0.0, 100.0)
    ) yield MseStats(w, mean, mse)
  )
}

class LazyStatsTest extends FlatSpec with TestHelpers with Matchers with GeneratorDrivenPropertyChecks {
  import LazyStatsTest._

  def shouldBeRoughlyEqual(actual: Double, expected: Double) {
    if (!expected.isInfinite && !expected.isNaN) {
      actual should be (roughlyEqualTo(expected)(tolerance))
    }
  }

  def shouldBeRoughlyEqual(actual: MseStats[Double], expected: MseStats[Double]) {
    shouldBeRoughlyEqual(actual.weight, expected.weight)
    shouldBeRoughlyEqual(actual.mean, expected.mean)
    shouldBeRoughlyEqual(actual.meanSquare, expected.meanSquare)
  }

  "LazyStats.mean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      shouldBeRoughlyEqual(LazyStats.mean(input), input.sum / input.size)
    }
  }

  "LazyStats.runningMean" should "be sum / size" in {
    forAll { (head: Double, tail: List[Double]) =>
      val input = head :: tail
      val means = LazyStats.runningMean(input)
      for ((m, i) <- means.zipWithIndex) {
        val expected = input.take(i).sum / math.max(i, 1)
        shouldBeRoughlyEqual(m, expected)
      }
    }
  }

  "WeightedMse.of" should "be weightedMean of squaredErrors" in {
    forAll { (head: Weighted[Double], tail: List[Weighted[Double]]) =>
      val input = head :: tail
      val actualMse = WeightedMse.of(input)
      // calculate MSE the obvious way
      val mean = LazyStats.weightedMean(input)
      val errors = input.map(_.map(x => { val diff = x - mean; diff * diff }))
      val expectedMse = LazyStats.weightedMean(errors)
      shouldBeRoughlyEqual(actualMse, expectedMse)
    }
  }

  "WeightedMse.Stats.plus" should "be commutative" in {
    forAll { (a: MseStats[Double], b: MseStats[Double]) =>
      shouldBeRoughlyEqual(a + b, b + a)
    }
  }

  it should "be associative" in {
    forAll { (a: MseStats[Double], b: MseStats[Double], c: MseStats[Double]) =>
      shouldBeRoughlyEqual(a + (b + c), (a + b) + c)
    }
  }

  "WeightedMse.Stats.minus" should "be inverse of plus" in {
    forAll { (b: MseStats[Double], a: MseStats[Double]) =>
      shouldBeRoughlyEqual(a, (a + b) - b)
    }
    forAll { (a: MseStats[Double], b: MseStats[Double]) =>
      shouldBeRoughlyEqual(a, (a - b) + b)
    }
  }
}
