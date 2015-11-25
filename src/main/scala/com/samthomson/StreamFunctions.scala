package com.samthomson

import scala.collection.immutable.Stream
import scala.language.implicitConversions


object StreamFunctions {
  def unfold[A, B](start: A)(step: A => Option[(B, A)]): Stream[B] = {
    step(start) match {
      case None  => Stream.empty
      case Some((b, a)) => b #:: unfold(a)(step)
    }
  }
}
