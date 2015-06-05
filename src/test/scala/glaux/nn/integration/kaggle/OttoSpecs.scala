package glaux.nn.integration.kaggle

import glaux.linalg.Dimension.{TwoD, Row}
import glaux.linalg.{Matrix, RowVector}
import glaux.nn.InputLayer
import glaux.nn.Net.SimpleNet
import glaux.nn.layers.{Softmax, Regression, FullyConnected}
import glaux.nn.trainers.BatchTrainer.{SGDOptions, MomentumSGDOptions, MomentumSGD}
import org.specs2.mutable.Specification

import scala.io.Source

class OttoSpecs extends Specification {
  val source = Source.fromURL(getClass.getResource("/kaggle/train.csv"))
  val pairs = source.getLines().drop(1).map { line =>
    val values = line.split(",")
    val inputValues = values.drop(1).dropRight(1).map(_.toDouble)
    val input = RowVector(inputValues:_*)
    val target = RowVector(values.takeRight(1).head.replace("Class_", "").toDouble)
    (input, target)
  }.toStream

  "read resource correctly" >> {
    pairs.length === 61878
    val allTargets = pairs.map(_._2)
    val allInputs = pairs.map(_._1)
    allTargets.map(_.head.toInt).distinct must contain(exactly(1,2,3,4,5,6,7,8,9))
    allTargets.map(_.dimension.totalSize).distinct must contain(exactly(1))
    allInputs.map(_.dimension.totalSize).distinct must be_==(Stream(93))
  }

  val numOfFeatures = 93
  val numOfClass = 9
  val dim: Row = Row(numOfFeatures)
  val inputLayer = InputLayer[RowVector](dim)
  val hiddenLayer = FullyConnected(Matrix.fill(TwoD(numOfFeatures, numOfClass), 0.5), RowVector.fill(Row(numOfClass), 0))
  val lossLayer = Softmax(Row(numOfClass))
  val initNet: SimpleNet[RowVector] = SimpleNet(inputLayer, Seq(hiddenLayer), lossLayer)

  val trainer = MomentumSGD[SimpleNet[RowVector]](MomentumSGDOptions(SGDOptions(learningRate = 0.01), 0.5))
  val initResult = trainer.init(initNet)



}
