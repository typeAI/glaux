package glaux.nn.trainers

import glaux.nn.Dimension.{TwoD, Row}
import glaux.nn.Net.SimpleNet
import glaux.nn.layers.{Regression, FullyConnected}
import glaux.nn.{Matrix, InputLayer, RowVector, Net}
import org.specs2.mutable.Specification

class VanillaSGDTrainerSpec extends Specification {

  val inputLayer = InputLayer[RowVector](Row(3))
  val hiddenLayer = FullyConnected(Matrix.uniform(TwoD(3, 1), 0.5), RowVector(1))
  val lossLayer = Regression(1)
  val initNet: Net[RowVector] = SimpleNet(inputLayer, Seq(hiddenLayer), lossLayer)



}
