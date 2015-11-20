package com.samthomson.data

import scala.language.higherKinds


object TraversableOnceOps extends Maxes

trait Maxes {
  /**
    * Provides analogs of `maxBy`, `max`, `minBy`, `min` that
    * return _all_ maxes/mins in the case of ties.
    */
  implicit class MaxesOps[A](xs: TraversableOnce[A]) {
    def maxesBy[C, B >: C](f: A => C)(implicit ord: Ordering[B]): List[A] = {
      val (bests, _) = xs.foldLeft((List.empty[A], Option.empty[C])) {
        case ((_, None), y) => (List(y), Some(f(y))) // y is the first and best so far
        case ((oldBests, someOldF @ Some(oldF)), y) =>
          val fy = f(y)
          ord.compare(fy, oldF) match {
            case 0 => (y :: oldBests, someOldF) // tied. add it to the list
            case cmp if cmp > 0 => (List(y), Some(fy)) // new best!
            case _ => (oldBests, someOldF) // try harder, y
          }
      }
      bests
    }

    def minsBy[C, B >: C](f: A => C)(implicit ord: Ordering[B]): List[A] = xs.maxesBy[C, B](f)(ord.reverse)

    def maxes[B >: A](implicit ord: Ordering[B]): List[A] = maxesBy[A, B](identity)(ord)

    def mins[B >: A](implicit  ord: Ordering[B]): List[A] = xs.maxes(ord.reverse)
  }
}

