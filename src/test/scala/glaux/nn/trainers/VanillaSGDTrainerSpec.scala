package glaux.nn.trainers

import glaux.linalg.{Matrix, RowVector, Dimension}
import Dimension.{TwoD, Row}
import glaux.nn.Net.DefaultNet
import glaux.nn.layers.{Regression, FullyConnected}
import glaux.nn.{InputLayer, Net}
import glaux.statistics
import org.specs2.mutable.Specification


class VanillaSGDTrainerSpec extends Specification {
  val dim: Row = Row(3)
  val inputLayer = InputLayer[RowVector](Row(3))
  val hiddenLayer = FullyConnected(Matrix(3, 1, Seq(0.5, -0.7, 1.5)), RowVector(0))
  val lossLayer = Regression(1)
  val initNet: DefaultNet[RowVector] = DefaultNet(inputLayer, Seq(hiddenLayer), lossLayer)

  val trainer = VanillaSGD[DefaultNet[RowVector]](SGDOptions(learningRate = 0.05))
  val initResult = trainer.init(initNet)

  val dist = statistics.distributions.normal(0, 3)
  val noise = statistics.distributions.normal(0, 0.01)
  def randomSample() : (initNet.Input, initNet.Output) ={
    val input = RowVector.sampleOf(dim, dist)
    val output = RowVector(input.sumAll + 1 + noise.sample)
    (input, output)
  }

  "net consistent with convnetjs" >> {
    val df = initNet.forward(RowVector(3,2,1))
    df.last.out must_== RowVector(1.6)
  }

  "trainer consistent with convnetjs" >> {
    val result = trainer.trainBatch(initResult)(Seq((RowVector(3,2,1), RowVector(5))))
    result.lossInfo.cost must beCloseTo(5.78 within 4.significantFigures)
  }

  "train summation" >> {
    val batches = 0.until(100).map(_ => 1.until(3).map(_ => randomSample() ))
    val finalResult = batches.foldLeft(initResult){ (lastResult, batch) =>
      trainer.trainBatch(lastResult)(batch)
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
  * layer_defs.push({type:'regression', num_neurons: 1});
  *
  * var net = new convnetjs.Net();
  * net.makeLayers(layer_defs);
  * var trainer = new convnetjs.Trainer(net, {method: 'sgd', learning_rate: 0.05,
                                    l2_decay: 0, momentum: 0, batch_size: 1,
                                    l1_decay: 1});
  * net.layers[1].filters[0].w = [0.5,-0.7,1.5]
  * //verify net output
  * net.forward(new convnetjs.Vol([3,2,1]));
  * trainer.train(new convnetjs.Vol([3,2,1]), 5)
  */
