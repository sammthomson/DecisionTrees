package com.samthomson

import org.scalactic.TripleEqualsSupport.Spread


trait TestHelpers {
  val tolerance = 0.00001

  /** Checks for scale-independent near-equality */
  def roughlyEqualTo(pivot: Double)(tolerance: Double): Spread[Double] = {
    if (math.abs(pivot) <= tolerance) {
      Spread(pivot, tolerance)
    } else {
      val Seq(min, max) = Seq(1.0 - tolerance, 1.0 + tolerance).map(pivot * _).sorted
      val center = (max + min) / 2.0
      val radius = max - center
      Spread(center, radius)
    }
  }
}
