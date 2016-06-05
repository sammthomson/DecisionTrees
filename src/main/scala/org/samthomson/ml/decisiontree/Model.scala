package org.samthomson.ml.decisiontree

import spire.algebra.AdditiveMonoid
import spire.implicits._

import scala.language.implicitConversions


@SerialVersionUID(1L)
trait Model[-X, +Y] {
  def predict(input: X): Y

  def apply(x: X): Y = predict(x)
}
object Model {
  def apply[X, Y](f: X => Y): Model[X, Y] = new Model[X, Y] {
    override def predict(input: X): Y = f(input)
  }
}

/**
  * A model that scores every possible output for an input, then predicts the highest scoring output.
  * Useful when each input has a tractable (e.g. not exponential) set of possible outputs.
  */
case class MultiClassModel[-X, Y](outputSpace: X => Iterable[Y],
                                  scoringModel: Model[(X, Y), Double]) extends Model[X, Y] with Equals {

  def scores(input: X): Iterable[(Y, Double)] = outputSpace(input).map(o => o -> scoringModel.predict((input, o)))

  override def predict(input: X): Y = scores(input).maxBy(_._2)._1

  /** Ignores `outputSpace` b/c functions are hard to test for equality */
  override def equals(other: Any): Boolean = other match {
    case that: MultiClassModel[X, Y] => (that canEqual this) && that.scoringModel == scoringModel
    case _ => false
  }
}
object MultiClassModel {
  def uniform[X, Y](outputSpace: X => Iterable[Y]): MultiClassModel[X, Y] =
    MultiClassModel(outputSpace, Leaf(0.0))
}

@SerialVersionUID(1L)
case class Ensemble[-X, +Y: AdditiveMonoid](baseModels: Vector[Model[X, Y]]) extends Model[X, Y] {
  override def predict(input: X): Y = baseModels.map(_.predict(input)).qsum
}
