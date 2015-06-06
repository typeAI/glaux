package glaux.nn.integration.kaggle

import glaux.linalg.Dimension.{TwoD, Row}
import glaux.linalg.{Matrix, RowVector}
import glaux.nn.InputLayer
import glaux.nn.Net.SimpleNet
import glaux.nn.layers.{Relu, Softmax, Regression, FullyConnected}
import glaux.nn.trainers.BatchTrainer.{SGDOptions, MomentumSGDOptions, MomentumSGD}
import org.specs2.mutable.Specification

import scala.io.Source
import scala.util.Random

class OttoSpecs extends Specification {

  "integration test with Otto data" >> skipped {
    val source = Source.fromURL(getClass.getResource("/kaggle/train.csv"))
    val pairs = source.getLines().drop(1).map { line =>
      val values = line.split(",")
      val inputValues = values.drop(1).dropRight(1).map(_.toDouble)
      val input = RowVector(inputValues: _*)
      val target = values.takeRight(1).head.replace("Class_", "").toInt
      (input, target)
    }.toStream

    "read resource correctly" >> {
      pairs.length === 61878
      val allTargets = pairs.map(_._2)
      val allInputs = pairs.map(_._1)
      allTargets.distinct must contain(exactly(1, 2, 3, 4, 5, 6, 7, 8, 9))
      allInputs.map(_.dimension.totalSize).distinct must be_==(Stream(93))
    }


    val numOfFeatures = 93
    val numOfMidNeurons = 45
    val numOfClass = 9
    val dim: Row = Row(numOfFeatures)
    val inputLayer = InputLayer[RowVector](dim)
    val fc1 = FullyConnected(Matrix.fill(TwoD(numOfFeatures, numOfMidNeurons), 0.1), RowVector.fill(Row(numOfMidNeurons), 0.1))
    val relu = Relu[RowVector](Row(numOfMidNeurons))
    val fc2 = FullyConnected(Matrix.fill(TwoD(numOfMidNeurons, numOfClass), 0.1), RowVector.fill(Row(numOfClass), 0))
    val lossLayer = Softmax(Row(numOfClass))
    val initNet: SimpleNet[RowVector] = SimpleNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)

    val trainer = MomentumSGD[SimpleNet[RowVector]](MomentumSGDOptions(SGDOptions(learningRate = 0.01), 0.9))
    val initResult = trainer.init(initNet)

    val batchSize = 5
    val batches = Random.shuffle(pairs).grouped(5) //two batches first

    def classificationVector(classification: Int, numOfClassification: Int) : RowVector = {
      val seqValues = (Seq.fill(classification - 1)(0) ++ Seq(1) ++ Seq.fill(9 - classification)(0)).map(_.toDouble)
      RowVector(seqValues: _*)
    }
    var track = 0
    val result = batches.foldLeft(initResult) { (lastResult, batch) =>
      val processedBatch = batch.map {
        case (input, target) => (input, classificationVector(target, 9))
      }
      track += 1
      println(s"training batch $track / ${61878/5}")
      println(s"last lost was ${lastResult.lossInfo.cost}")
      trainer.trainBatch(processedBatch, lastResult)

    }

    "result" >> {
      val sample1 = pairs(1)._1
      val sample6 = pairs(34483)._1
      val sample8 = pairs(48668)._1
      def classification(vector: RowVector): Int = {
        val max = vector.seqView.max
        val maximized = vector.map( (v:Double) => if(v < max) 0 else 1)
        (maximized dot RowVector(1, 2, 3, 4, 5, 6, 7, 8, 9)).toInt
      }

      classification(result.net.predict(sample1)) === 1
      classification(result.net.predict(sample6)) === 6
      classification(result.net.predict(sample8)) === 8
    }

    1 === 1
  }
}
