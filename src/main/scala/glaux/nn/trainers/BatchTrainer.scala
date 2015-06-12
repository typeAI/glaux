package glaux.nn.trainers

import glaux.linalg._
import glaux.nn._


trait BatchTrainer[Trainee <: Net] {
  type Input = Trainee#Input
  type Output = Trainee#Output

  type CalculationContext

  case class BatchResult(lossInfo: LossInfo, net: Trainee, batchSize: Int, calcContext: CalculationContext)

  val build: Net.CanBuildFrom[Trainee]

  def initialCalculationContext(net: Trainee) : CalculationContext

  val initialLossInfo = LossInfo(0,0,0)
  def init(net: Trainee) = BatchResult(initialLossInfo, net, 0, initialCalculationContext(net))

  type ScalarOutputInfo = (Double, Int) //(Value, Index)

  //training based on only partial correction about output - a scalar value at an index, this helps us with regression on a single scalar value
  def trainBatchWithScalaOutputInfo(batch: Iterable[(Input, ScalarOutputInfo)], lastResult: BatchResult) : BatchResult = {
    val net: Trainee = lastResult.net
    val dataFlowBatch = batch.map {
      case (input, scalarOutputInfo) => {
        val dataFlow = net.forward(input.asInstanceOf[net.Input])
        val (targetScalar, index) = scalarOutputInfo
        (dataFlow, dataFlow.last.out.update(index, targetScalar))
      }
    }
    trainBatchWithFullDataFlow(dataFlowBatch, lastResult)
  }

  def trainBatch(batch: Iterable[(Input, Output)], lastResult: BatchResult): BatchResult = {
    val net: Trainee = lastResult.net
    val dataFlowBatch = batch.map {
      case (input, output) => (net.forward(input.asInstanceOf[net.Input]), output)
    }
    trainBatchWithFullDataFlow(dataFlowBatch, lastResult)
  }

  private def trainBatchWithFullDataFlow(batch: Iterable[(DataFlow, Output)], lastResult: BatchResult): BatchResult = {
    val net: Trainee = lastResult.net
    val pairs = batch.map {
      case (dataFlow, target) => net.backward(target, dataFlow)
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
