package org.samthomson

import org.scalactic.TripleEqualsSupport.Spread


trait TestHelpers {
  val tolerance = 0.00001

  /** Checks for scale-independent near-equality */
  def roughlyEqualTo(pivot: Double)(tolerance: Double): Spread[Double] = {
    if (math.abs(pivot) <= tolerance) {
      Spread(pivot, tolerance)
    } else if (pivot == Double.PositiveInfinity || pivot == Double.NegativeInfinity) {
      Spread(pivot, tolerance)
    } else {
      val Seq(min, max) = Seq(1.0 - tolerance, 1.0 + tolerance).map(pivot * _).sorted
      val radius = max / 2.0 - min / 2.0
      val center = min + radius
      Spread(center, radius)
    }
  }
}
