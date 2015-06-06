package glaux.nn.trainers

import glaux.linalg.{Matrix, RowVector, Dimension}
import Dimension.{TwoD, Row}
import glaux.nn.Net.SimpleNet
import glaux.nn.layers.{Regression, FullyConnected}
import glaux.nn.trainers.BatchTrainer.{SGDOptions, VanillaSGD}
import glaux.nn.{InputLayer, Net}
import glaux.statistics
import org.specs2.mutable.Specification


class VanillaSGDTrainerSpec extends Specification {
  val dim: Row = Row(3)
  val inputLayer = InputLayer[RowVector](Row(3))
  val hiddenLayer = FullyConnected(Matrix.fill(TwoD(3, 1), 0), RowVector(0))
  val lossLayer = Regression(1)
  val initNet: SimpleNet[RowVector] = SimpleNet(inputLayer, Seq(hiddenLayer), lossLayer)

  val trainer = VanillaSGD[SimpleNet[RowVector]](SGDOptions(learningRate = 0.05))
  val initResult = trainer.init(initNet)

  val dist = statistics.distributions.normal(0, 3)
  val noise = statistics.distributions.normal(0, 0.01)
  def randomSample() : (initNet.Input, initNet.Output) ={
    val input = RowVector.sampleOf(dim, dist)
    val output = RowVector(input.sumAll + 1 + noise.sample)
    (input, output)
  }

  "train summation" >> {
    val batches = 0.until(100).map(_ => 1.until(3).map(_ => randomSample() ))
    val finalResult = batches.foldLeft(initResult){ (lastResult, batch) =>
      trainer.trainBatch(batch, lastResult)
    }

    val result = finalResult.net.predict(RowVector(2,3,4))
    result.dimension.size === 1
    result(0) must  beCloseTo(10.0 within 3.significantFigures)
  }

}

/** convnetjs test code
  *
  * var layer_defs = [];
  * layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:3});
  * layer_defs.push({type:'fc', in_sx:1, in_sy:1, in_depth:3, num_neurons: 1});
  * layer_defs.push({type:'regression', num_neurons: 1});
  *
  * var net = new convnetjs.Net();
  * net.makeLayers(layer_defs);
  * var trainer = new convnetjs.Trainer(net, {method: 'sgd', learning_rate: 0.05,
                                    l2_decay: 0, momentum: 0, batch_size: 3,
                                    l1_decay: 0});

  *
  */
