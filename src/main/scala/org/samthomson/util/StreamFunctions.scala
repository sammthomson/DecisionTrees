package org.samthomson.util

import scala.collection.immutable.Stream

object StreamFunctions {
  def unfold[A, B](start: A)(step: A => Option[(B, A)]): Stream[B] = {
    step(start) match {
      case None  => Stream.empty
      case Some((b, a)) => b #:: unfold(a)(step)
    }
  }
}
