package com.samthomson.ml.decisiontree

import spire.algebra.AdditiveMonoid
import spire.implicits._

import scala.language.implicitConversions


@SerialVersionUID(1L)
trait Model[-X, +Y] extends Function[X, Y] {
  def predict(input: X): Y

  override def apply(x: X): Y = predict(x)
}

/** a model with a fixed finite set of possible outputs */
case class MultiClassModel[-X, Y](outputSpace: Iterable[Y],
                                  scoringModel: Model[(X, Y), Double]) extends Model[X, Y] {

  def scores(input: X): Iterable[(Y, Double)] = outputSpace.map(o => o -> scoringModel.predict((input, o)))

  override def predict(input: X): Y = scores(input).maxBy(_._2)._1
}

case class Ensemble[-X, Y: AdditiveMonoid](baseModels: Iterable[Model[X, Y]]) extends Model[X, Y] {
  override def predict(input: X): Y = baseModels.map(_.predict(input)).qsum
}
