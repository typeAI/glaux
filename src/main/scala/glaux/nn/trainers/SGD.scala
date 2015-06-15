package glaux.nn.trainers

import glaux.linalg.Tensor
import glaux.nn._


case class SGDOptions(learningRate: Double = 0.01, l1Decay: Decay = 0, l2Decay: Decay = 0)

abstract class SGDBase[NT <: Net: Net.CanBuildFrom](options: SGDOptions) extends BatchTrainer[NT] {
  type Trainee = NT

  case class NewParamResult(newParam: LayerParam, l1DecayLoss: Loss, l2DecayLoss: Loss, adjustment: Tensor)
  type Results =  Map[HiddenLayer, Seq[NewParamResult]]

  val build: Net.CanBuildFrom[Trainee] = implicitly[Net.CanBuildFrom[Trainee]]

  def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Tensor, lastContext: CalculationContext) : Tensor

  def calculate(netParamGradients: NetParamGradients, lastIterationResult: BatchResult, loss: Loss, batchSize: Int): (NetParams, CalculationContext, LossInfo) = {
    val lastContext = lastIterationResult.calcContext


    def calcNewParam(paramGrad: ParamGradient, layer: HiddenLayer): NewParamResult = {
      val l1Decay =  options.l1Decay * paramGrad.param.regularizationSetting.l1DM
      val l2Decay =  options.l2Decay * paramGrad.param.regularizationSetting.l2DM
      val p2DL: Tensor = (paramGrad.param.value * paramGrad.param.value) * l2Decay / 2
      val p1DL: Tensor = paramGrad.param.value.map(Math.abs(_)) * l1Decay
      val l2DecayLoss: Loss = p2DL.sumAll
      val l1DecayLoss: Loss = p1DL.sumAll
      val param = paramGrad.param
      val pValue = param.value
      val l1grad: Tensor = pValue.map((v: Double) => if(v > 0) 1 else -1) * l1Decay
      val l2grad: Tensor = pValue * l2Decay
      val rawBatchGradient: Tensor = (l1grad + l2grad + paramGrad.value) / batchSize

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

  def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Tensor, lastContext: Unit): Tensor =
    rawBatchGradient *  (- options.learningRate)

}

case class MomentumSGDOptions(sgdOptions: SGDOptions, momentum: Double)
case class MomentumSGD[ NT <: Net: Net.CanBuildFrom](options: MomentumSGDOptions) extends SGDBase[NT](options.sgdOptions) {

  case class ParamGSum(layer: HiddenLayer, param: LayerParam, value: Tensor)

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


  def calculateParamAdjustment(layer: HiddenLayer, param: LayerParam, rawBatchGradient: Tensor, lastContext: MomentumSGDIterationContext): Tensor = {
    val lastGsum = lastContext.gSums.find(gs => gs.param.id == param.id && gs.layer.id == layer.id).get
    (lastGsum.value * options.momentum) - (rawBatchGradient * options.sgdOptions.learningRate)
  }

}
