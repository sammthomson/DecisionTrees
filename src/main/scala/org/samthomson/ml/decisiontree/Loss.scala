package org.samthomson.ml.decisiontree

import scala.math.{exp, log}


/**
  * @tparam Y type of the gold output
  * @tparam P type of predictions (e.g. `Double` for regression, `Y => Double` for discrete Y)
  */
trait Loss[-Y, -P] {
  def loss(gold: Y)(predicted: P): Double
}

trait DiffableLoss[-Y, P] extends Loss[Y, P] {
  def lossAndGradient(gold: Y)(predicted: P): (Double, P)
  // derived
  override def loss(gold: Y)(predicted: P): Double = lossAndGradient(gold)(predicted)._1
}

trait TwiceDiffableLoss[-Y, P] extends DiffableLoss[Y, P] {
  // TODO: if `P` is multivariate (as it is in structured prediction), this is
  // actually just a diagonalized approximation of hessian. Does that matter?
  // I.e. if `P` is `Y => Double`, hessian should really be `(Y, Y) => Double`.
  def lossGradAndHessian(gold: Y)(predicted: P): (Double, P, P)
  // derived
  override def lossAndGradient(gold: Y)(predicted: P): (Double, P) = {
    val (loss, grad, _) = lossGradAndHessian(gold)(predicted)
    (loss, grad)
  }
}

object TwiceDiffableLoss {
  /** (gold - predicted)^2^ */
  object SquaredError extends TwiceDiffableLoss[Double, Double] {
    override def lossGradAndHessian(gold: Double)(predicted: Double): (Double, Double, Double) = {
      val err = gold - predicted
      (err * err, 2.0 * err, 2.0)
    }
  }

  /** -ln( e^score(gold)^ / sum_y { e^score(y)^ } ) */
  case class MultiClassLogLoss[Y]() extends TwiceDiffableLoss[Y, Map[Y, Double]] {
    override def lossGradAndHessian(gold: Y)(predicted: Map[Y, Double]): (Double, Map[Y, Double], Map[Y, Double]) = {
      val exponentiatedScores = predicted.mapValues(exp)
      val partition = 1.0 / exponentiatedScores.values.sum
      val predictedProbs = exponentiatedScores.mapValues(_ * partition)
      val goldProb = predictedProbs(gold)
      val loss = -log(goldProb)
      val gradient = predictedProbs.updated(gold, goldProb - 1.0)
      val hessian = predictedProbs.mapValues(p => p * (1 - p))
      (loss, gradient, hessian)
    }
  }

  /** sum_y { max(0, score(y) + cost(y) - score(gold)) } */
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

  /** sum_y { max(0, score(y) + cost(y) - score(gold))^2^ } */
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
        val loss = violations.values.map(s => s * s).sum
        val grads =
          violations.mapValues(2.0 * _) ++
              safe.mapValues(_ => 0.0) +
              (gold -> -2.0 * violations.values.sum)
        val hess =
          marginViolators.mapValues(_ => 2.0) ++
              safe.mapValues(_ => 0.0) +
              (gold -> 2.0 * marginViolators.size.toDouble) // 2nd deriv is always non-negative
        (loss, grads, hess)
      }
    }
  }
}
