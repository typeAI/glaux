package glaux.nn.trainers

import glaux.linalg.Dimension.{Row, TwoD}
import glaux.linalg.{Dimension, Matrix, RowVector}
import glaux.nn.InputLayer
import glaux.nn.Net.DefaultNet
import glaux.nn.layers.{FullyConnected, Regression}
import glaux.nn.trainers.MomentumSGD.Settings
import glaux.nn.trainers.SGD.SGDSettings
import glaux.statistics
import org.specs2.mutable.Specification


class MomentumSGDTrainerSpec extends Specification {

  val dim: Row = Row(3)
  val inputLayer = InputLayer[RowVector](dim)
  val hiddenLayer = FullyConnected(Matrix(3, 1, Seq(0.5, -0.7, 1.5)), RowVector(0))
  val lossLayer = Regression(1)
  val initNet: DefaultNet[RowVector] = DefaultNet(inputLayer, Seq(hiddenLayer), lossLayer)

  val trainer = MomentumSGD[DefaultNet[RowVector]](Settings(SGDSettings(learningRate = 0.01), 0.9))
  val initResult = trainer.init(initNet)

  "init context" >> {
    initResult.calcContext.gSums.size === hiddenLayer.params.size
    initResult.calcContext.gSums.find(gs => gs.param.id == hiddenLayer.params.head.id && gs.layer == hiddenLayer) must beSome[MomentumSGD.ParamGSum]
  }

  "net consistent with convnetjs" >> {
    val df = initNet.forward(RowVector(3,2,1))
    df.last.out must_== RowVector(1.6)
  }

  "trainer consistent with convnetjs" >> {
    val result = trainer.trainBatch(initResult)(Seq((RowVector(3,2,1), RowVector(7))))
    result.lossInfo.cost must beCloseTo(14.58 within 4.significantFigures)

    result.calcContext.gSums.head.value must_== Matrix(3, 1, Seq(0.162, 0.108, 0.054))
    result.calcContext.gSums.last.value must_== RowVector(0.054)


    val result2 = trainer.trainBatch(result)(Seq((RowVector(4,4,5), RowVector(14))))
    result2.lossInfo.cost must beCloseTo(17.381408 within 4.significantFigures)

    val result3 = trainer.trainBatch(result2)(Seq((RowVector(-1,1,3), RowVector(4))))
    result3.lossInfo.cost must beCloseTo(0.15188 within 4.significantFigures)
  }

  val dist = statistics.distributions.normal(0, 3)
  val noise = statistics.distributions.normal(0, 0.001)
  def randomSample() : (initNet.Input, initNet.Output) ={
    val input = RowVector.sampleOf(dim, dist)
    val output = RowVector(input.sumAll + 1 + noise.sample)
    (input, output)
  }

  "train summation" >> {
    val batches = 0.until(100).map(_ => 1.until(3).map(_ => randomSample() ))
    val finalResult = batches.foldLeft(initResult){ (lastResult, batch) =>
      trainer.trainBatch(lastResult)(batch)
    }

    val result = finalResult.net.predict(RowVector(2,3,4))
    result.dimension.size === 1
    result(0) must beCloseTo(10.0 within 2.significantFigures)
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
  * var trainer = new convnetjs.Trainer(net, {method: 'sgd', learning_rate: 0.01,
                                    l2_decay: 0, momentum: 0.9, batch_size: 1,
                                    l1_decay: 1});
  * net.layers[1].filters[0].w = [0.5,-0.7,1.5]
  *
  * trainer.train(new convnetjs.Vol([3,2,1]), 7)
  * trainer.train(new convnetjs.Vol([4,4,5]), 14)
  * trainer.train(new convnetjs.Vol([-1,1,3]), 4)
  */
