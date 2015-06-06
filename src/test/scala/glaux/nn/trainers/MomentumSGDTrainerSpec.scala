package glaux.nn.trainers

import glaux.linalg.Dimension.{Row, TwoD}
import glaux.linalg.{Dimension, Matrix, RowVector}
import glaux.nn.InputLayer
import glaux.nn.Net.SimpleNet
import glaux.nn.layers.{FullyConnected, Regression}
import glaux.nn.trainers.BatchTrainer.{MomentumSGDOptions, MomentumSGD, SGDOptions, VanillaSGD}
import glaux.statistics
import org.specs2.mutable.Specification


class MomentumSGDTrainerSpec extends Specification {

  val dim: Row = Row(3)
  val inputLayer = InputLayer[RowVector](dim)
  val hiddenLayer = FullyConnected(Matrix.fill(TwoD(3, 1), 0.1), RowVector(0))
  val lossLayer = Regression(1)
  val initNet: SimpleNet[RowVector] = SimpleNet(inputLayer, Seq(hiddenLayer), lossLayer)

  val trainer = MomentumSGD[SimpleNet[RowVector]](MomentumSGDOptions(SGDOptions(learningRate = 0.01), 0.8))
  val initResult = trainer.init(initNet)

  "init context" >> {
    initResult.calcContext.gSums.size === hiddenLayer.params.size
    initResult.calcContext.gSums.find(gs => gs.param.id == hiddenLayer.params.head.id && gs.layer == hiddenLayer) must beSome[trainer.ParamGSum]
  }

  val dist = statistics.distributions.normal(0, 3)
  val noise = statistics.distributions.normal(0, 0.01)
  def randomSample() : (initNet.Input, initNet.Output) ={
    val input = RowVector.sampleOf(dim, dist)
    val output = RowVector(input.sumAll + 1 + noise.sample)
    (input, output)
  }

  "train summation" >> {
    val batches = 0.until(150).map(_ => 1.until(3).map(_ => randomSample() ))
    val finalResult = batches.foldLeft(initResult){ (lastResult, batch) =>
      trainer.trainBatch(batch, lastResult)
    }

    val result = finalResult.net.predict(RowVector(2,3,4))
    result.dimension.size === 1
    result(0) must  beCloseTo(10.0 within 3.significantFigures)
  }

}