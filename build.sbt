name := "DecisionTrees"

version := "0.1-SNAPSHOT"

organization := "com.samthomson"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.spire-math" %% "spire" % "0.11.0",
  // logging
  "ch.qos.logback" %  "logback-classic" % "1.1.7",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
  // test
  "org.scalatest" %% "scalatest" % "2.2.2" % "test",
  "org.scalacheck" %% "scalacheck" % "1.12.2" % "test"
)

fork in Test := true
