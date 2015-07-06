package glaux.nn.integration.kaggle

import glaux.linalg.Dimension.{TwoD, Row}
import glaux.linalg.{Matrix, RowVector}
import glaux.nn.InputLayer
import glaux.nn.Net.DefaultNet
import glaux.nn.layers.{Relu, Softmax, FullyConnected}
import glaux.nn.trainers.{SGDOptions, MomentumSGDOptions, MomentumSGD}
import org.specs2.mutable.Specification

import scala.io.Source
import scala.util.Random

class OttoSpecs extends Specification {
  val runIntegration = false //todo make this configurable
  if(runIntegration) {
    "integration test with Otto data" >> {
      val source = Source.fromURL(getClass.getResource("/kaggle/train.csv"))
      val pairs = source.getLines().drop(1).map { line =>
        val values = line.split(",")
        val inputValues = values.drop(1).dropRight(1).map(_.toDouble)
        val input = RowVector(inputValues: _*)
        val target = values.takeRight(1).head.replace("Class_", "").toInt
        (input, target)
      }.toSeq

      val (trainning, test) = Random.shuffle(pairs).splitAt(50000)

      "read resource correctly" >> {
        pairs.length === 61878
        val allTargets = pairs.map(_._2)
        val allInputs = pairs.map(_._1)
        allTargets.distinct must contain(exactly(1, 2, 3, 4, 5, 6, 7, 8, 9))
        allInputs.map(_.dimension.totalSize).distinct must be_==(Seq(93))
      }

      val numOfFeatures = 93
      val numOfMidNeurons = 45
      val numOfClass = 9
      val dim: Row = Row(numOfFeatures)
      val inputLayer = InputLayer[RowVector](dim)
      import glaux.statistics.distributions.normal
      val fc1 = FullyConnected(Matrix.sampleOf(TwoD(numOfFeatures, numOfMidNeurons), normal(0, 2)), RowVector.fill(Row(numOfMidNeurons), 0.1))
      val relu = Relu[RowVector](Row(numOfMidNeurons))
      val fc2 = FullyConnected(Matrix.sampleOf(TwoD(numOfMidNeurons, numOfClass), normal(0, 2)), RowVector.fill(Row(numOfClass), 0))
      val lossLayer = Softmax(Row(numOfClass))
      val initNet: DefaultNet[RowVector] = DefaultNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)

      val trainer = MomentumSGD[DefaultNet[RowVector]](MomentumSGDOptions(SGDOptions(learningRate = 0.001), 0.9))
      val initResult = trainer.init(initNet)

      val batchSize: Int = 20
      val batches = trainning.grouped(batchSize) //two batches first

      def classificationVector(classification: Int, numOfClassification: Int): RowVector = {
        val seqValues = (Seq.fill(classification - 1)(0) ++ Seq(1) ++ Seq.fill(9 - classification)(0)).map(_.toDouble)
        RowVector(seqValues: _*)
      }
      var track = 0
      val result = batches.foldLeft(initResult) { (lastResult, batch) =>
        val processedBatch = batch.map {
          case (input, target) => (input, classificationVector(target, 9))
        }
        track += 1
        println(s"training batch $track / ${trainning.length / batchSize}")
        println(s"last lost was ${lastResult.lossInfo.cost}")
        trainer.trainBatch(lastResult)(processedBatch)
      }

      "reach 65% correct rate on test data" >> {
        def classification(vector: RowVector): Int = {
          val max = vector.seqView.max
          val maximized = vector.map((v: Double) => if (v < max) 0 else 1)
          (maximized dot RowVector(1, 2, 3, 4, 5, 6, 7, 8, 9)).toInt
        }
        val predictions: Iterable[Boolean] = test.map {
          case (input, target: Int) =>
            val prediction = classification(result.net.predict(input))
            prediction == target
        }

        predictions.count((b: Boolean) => b).toDouble / test.length must be_>(0.65)
      }

    }
  }
}
