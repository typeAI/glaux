package glaux.nn.trainers

import glaux.linalg._
import glaux.nn._

import scala.concurrent.Future

trait BatchTrainer[Trainee <: Net] {
  type Input = Trainee#Input
  type Output = Trainee#Output

  type CalculationContext
  
  case class BatchResult(lossInfo: LossInfo, net: Trainee, batchSize: Int, calcContext: CalculationContext)

  val build: Net.CanBuildFrom[Trainee]

  def initialCalculationContext(net: Trainee) : CalculationContext

  val initialLossInfo = LossInfo(0,0,0)
  def init(net: Trainee) = BatchResult(initialLossInfo, net, 0, initialCalculationContext(net))


  def trainBatch(batch: Iterable[(Input, Output)], lastResult: BatchResult): BatchResult = {
    val net: Trainee = lastResult.net
    val pairs = batch.map {
      case (input, target) =>
        val dataFlow = net.forward(input.asInstanceOf[net.Input]) //unforutnately this is the cleanest way to encode Type
        net.backward(target, dataFlow)
    }
    val batchLoss = pairs.last._1 //todo: confirm on this
    val batchSize = pairs.size
    val paramsGrads = accumulate(pairs.map(_._2))
    val (newParams, newContext, lossInfo) = calculate(paramsGrads, lastResult, batchLoss, batchSize)

    val newLayers = newParams.map {
      case (l, ps) => l.updateParams(ps)
    }
    val newLayersSorted = net.hiddenLayers.map { oldLayer =>
      newLayers.find(_.id == oldLayer.id).getOrElse(oldLayer) //it is possible that some layer didn't get updated and thus use the old layer instead
    }

    val newNet: Trainee = build(net, newLayersSorted)
    BatchResult(lossInfo, newNet, batchSize, newContext)
  }

  def accumulate(netParamGradients: Iterable[NetParamGradients]): NetParamGradients =
    netParamGradients.reduce { (npg1, npg2) =>
      (npg1, npg2).zipped.map {
        case ((layer1, paramGradients1),(layer2, paramGradients2)) =>
          assert(layer1 == layer2)//assume sequence of the two NetParamGradients are the same, but assert the match here
          val newParamGradient = (paramGradients1, paramGradients2).zipped.map { (paramGradient1, paramGradient2) =>
            assert(paramGradient1.param == paramGradient2.param) //assume the sequence remain the same, but assert the match here
            paramGradient1.copy(value = paramGradient1.value + paramGradient2.value)
          }
          (layer1, newParamGradient)
      }
    }


  def calculate(paramGrads: NetParamGradients, lastIterationResult: BatchResult, loss: Loss, batchSize: Int ): (NetParams, CalculationContext, LossInfo)
}


case class LossInfo(l1Decay: Loss, l2Decay: Loss, cost: Loss) {
  val total: Loss = l1Decay + l2Decay + cost
}

object BatchTrainer {

  case class SGDOptions(learningRate: Double = 0.01, l1Decay: Decay = 0, l2Decay: Decay = 0)

  abstract class SGDBase[NT <: Net: Net.CanBuildFrom](options: SGDOptions) extends BatchTrainer[NT] {
    type Trainee = NT


    case class NewParamResult(newParam: LayerParam, l1DecayLoss: Loss, l2DecayLoss: Loss, adjustment: Vol)
    type Results =  Map[HiddenLayer, Seq[NewParamResult]]

    val build: Net.CanBuildFrom[Trainee] = implicitly[Net.CanBuildFrom[Trainee]]

    def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Vol, lastContext: CalculationContext) : Vol

    def calculate(netParamGradients: NetParamGradients, lastIterationResult: BatchResult, loss: Loss, batchSize: Int): (NetParams, CalculationContext, LossInfo) = {
      val lastContext = lastIterationResult.calcContext


      def calcNewParam(paramGrad: ParamGradient, layer: HiddenLayer): NewParamResult = {
        val l1Decay =  options.l1Decay * paramGrad.param.regularizationSetting.l1DM
        val l2Decay =  options.l2Decay * paramGrad.param.regularizationSetting.l2DM
        val p2DL: Vol = (paramGrad.param.value * paramGrad.param.value) * l2Decay / 2
        val p1DL: Vol = paramGrad.param.value.map(Math.abs(_)) * l1Decay
        val l2DecayLoss: Loss = p2DL.sumAll
        val l1DecayLoss: Loss = p1DL.sumAll
        val param = paramGrad.param
        val pValue = param.value
        val l1grad: Vol = pValue.map((v: Double) => if(v > 0) 1 else -1) * l1Decay
        val l2grad: Vol = pValue * l2Decay
        val rawBatchGradient: Vol = (l1grad + l2grad + paramGrad.value) / batchSize

        val paramAdjustment = calculateParamAdjustment(layer, param, rawBatchGradient, lastContext)
        val newParam = param.copy(value = pValue + paramAdjustment)

        NewParamResult(newParam, l1DecayLoss, l2DecayLoss, paramAdjustment)
      }

      //note here if there is no paramGrad, the layer won't be calculated and will be missing
      val results: Results = (for {
        (layer, paramGrads) <- netParamGradients.toSeq
        paramGrad <- paramGrads
        newParamResult = calcNewParam(paramGrad, layer)
      } yield (layer, newParamResult)).groupBy(_._1).mapValues(_.map(_._2))

      val newNetParams: NetParams = results.mapValues(_.map(_.newParam))
      
      val newContext = updateContext(lastContext, results)
      val l1DecayLoss = results.values.flatten.map(_.l1DecayLoss).sum
      val l2DecayLoss = results.values.flatten.map(_.l2DecayLoss).sum
      val lossInfo = LossInfo(l1DecayLoss, l2DecayLoss, loss)
      (newNetParams, newContext, lossInfo)
    }

    def updateContext(lastContext: CalculationContext, results: Results): CalculationContext
  }

  case class VanillaSGD[NT <: Net: Net.CanBuildFrom](options: SGDOptions) extends SGDBase[NT](options) {

    type CalculationContext = Unit

    def initialCalculationContext(net: Trainee): Unit = ()

    def updateContext(lastContext: Unit, results: Results) = ()

    def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Vol, lastContext: Unit): Vol =
      rawBatchGradient *  (- options.learningRate)

  }

  case class MomentumSGDOptions(sgdOptions: SGDOptions, momentum: Double)
  case class MomentumSGD[ NT <: Net: Net.CanBuildFrom](options: MomentumSGDOptions) extends SGDBase[NT](options.sgdOptions) {

    case class ParamGSum(layer: HiddenLayer, param: LayerParam, value: Vol)

    case class MomentumSGDIterationContext(gSums: Seq[ParamGSum])

    type CalculationContext = MomentumSGDIterationContext

    def initialCalculationContext(net: Trainee): MomentumSGDIterationContext = {
      val paramGSums = net.hiddenLayers.flatMap { layer =>
        layer.params.map { p => ParamGSum(layer, p, p.value.fill(0))
        }
      }
      MomentumSGDIterationContext(paramGSums)
    }

    def updateContext(lastContext: MomentumSGDIterationContext, results: Results): MomentumSGDIterationContext = {
      val gsums = results.flatMap {
        case (layer, paramResults) => paramResults.map( pr => ParamGSum(layer, pr.newParam, pr.adjustment) )
      }.toSeq
      lastContext.copy(gSums = gsums)
    }


    def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Vol, lastContext: MomentumSGDIterationContext): Vol = {
      val lastGsum = lastContext.gSums.find(gs => gs.param.id == param.id && gs.layer.id == layer.id).get
      (lastGsum.value * options.momentum) - (rawBatchGradient * options.sgdOptions.learningRate)
    }
      
  }

}