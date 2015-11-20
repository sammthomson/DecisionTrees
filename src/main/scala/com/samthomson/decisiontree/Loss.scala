package com.samthomson.decisiontree


trait Loss[Y, P] {
  def loss(gold: Y)(predicted: P): Double
}

trait DiffableLoss[Y, P] extends Loss[Y, P] {
  def lossAndGradient(gold: Y)(predicted: P): (Double, P)
  // derived
  override def loss(gold: Y)(predicted: P): Double = lossAndGradient(gold)(predicted)._1
}

trait TwiceDiffableLoss[Y, P] extends DiffableLoss[Y, P] {
  def lossGradAndHessian(gold: Y)(predicted: P): (Double, P, P)
  // derived
  override def lossAndGradient(gold: Y)(predicted: P): (Double, P) = {
    val (loss, grad, _) = lossGradAndHessian(gold)(predicted)
    (loss, grad)
  }
}

object TwiceDiffableLoss {
  object SquaredError extends TwiceDiffableLoss[Double, Double] {
    override def lossGradAndHessian(gold: Double)(predicted: Double): (Double, Double, Double) = {
      val err = gold - predicted
      (err * err, 2.0 * err, 2.0)
    }
  }

  case class MultiClassHinge[Y]() extends TwiceDiffableLoss[Y, Map[Y, Double]] {
    def cost(gold: Y)(predicted: Y): Double = if (predicted == gold) 0.0 else 1.0

    override def lossGradAndHessian(gold: Y)(predicted: Map[Y, Double]): (Double, Map[Y, Double], Map[Y, Double]) = {
      val goldScore = predicted(gold)
      val augmented = predicted.map({ case (y, score) => y -> (score + cost(gold)(y)) })
      val (marginViolators, safe) = augmented.partition({ case (y, s) => s > goldScore})
      val zeros = predicted.mapValues(_ => 0.0)
      if (marginViolators.isEmpty) {
        (0.0, zeros, zeros) // no loss
      } else {
        val loss = marginViolators.values.map(_ - goldScore).sum
        val grads =
          marginViolators.mapValues(_ => 1.0) ++
              safe.mapValues(_ => 0.0) +
              (gold -> -marginViolators.size.toDouble)
        (loss, grads, zeros) // 2nd deriv is always 0
      }
    }
  }

  case class MultiClassSquaredHinge[Y]() extends TwiceDiffableLoss[Y, Map[Y, Double]] {
    val hinge = MultiClassHinge[Y]()

    override def lossGradAndHessian(gold: Y)(predicted: Map[Y, Double]): (Double, Map[Y, Double], Map[Y, Double]) = {
      val goldScore = predicted(gold)
      val costAugmented = predicted.map({ case (y, score) => y -> (score + hinge.cost(gold)(y)) })
      val (marginViolators, safe) = costAugmented.partition({ case (y, s) => s > goldScore})
      val zeros = predicted.mapValues(_ => 0.0)
      if (marginViolators.isEmpty) {
        (0.0, zeros, zeros) // no loss
      } else {
        val violations = marginViolators.mapValues(_ - goldScore)
        val loss = violations.values.map(s => .5 * s * s).sum
        val grads =
          violations ++
              safe.mapValues(_ => 0.0) +
              (gold -> -violations.values.sum)
        val hess =
          marginViolators.mapValues(_ => 1.0) ++
              safe.mapValues(_ => 0.0) +
              (gold -> marginViolators.size.toDouble) // 2nd deriv is always non-negative
        (loss, grads, hess)
      }
    }
  }
}
