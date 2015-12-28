package glaux.neuralnetwork.trainers

import glaux.neuralnetwork._
import glaux.neuralnetwork.trainers.BatchTrainer.LossInfo

trait BatchTrainer[Trainee <: Net, CalculationContext] {
  type Output = Trainee#Output

  val build: Net.Updater[Trainee]

  type BatchResult = BatchTrainer.BatchResult[Trainee, CalculationContext]

  def initialCalculationContext(net: Trainee): CalculationContext

  val initialLossInfo = LossInfo(0, 0, 0)
  def init(net: Trainee): BatchResult = BatchTrainer.BatchResult(initialLossInfo, net, 0, initialCalculationContext(net))

  type ScalarOutputInfo = (Double, Int) //(Value, Index)

  //training based on only partial correction about output - a scalar value at an index, this helps us with regression on a single scalar value
  def trainBatchWithScalaOutputInfo(lastResult: BatchResult)(batch: Iterable[(lastResult.net.Input, ScalarOutputInfo)]): BatchResult = {
    val dataFlowBatch = batch.map {
      case (input, scalarOutputInfo) ⇒ {
        val dataFlow = lastResult.net.forward(input)
        val (targetScalar, index) = scalarOutputInfo
        (dataFlow, dataFlow.last.out.update(index, targetScalar))
      }
    }
    trainBatchWithFullDataFlow(dataFlowBatch, lastResult)
  }

  def trainBatch(lastResult: BatchResult)(batch: Iterable[(lastResult.net.Input, Output)]): BatchResult = {
    val dataFlowBatch = batch.map {
      case (input, output) ⇒ (lastResult.net.forward(input), output)
    }
    trainBatchWithFullDataFlow(dataFlowBatch, lastResult)
  }

  private def trainBatchWithFullDataFlow(batch: Iterable[(DataFlow, Output)], lastResult: BatchResult): BatchResult = {
    val net: Trainee = lastResult.net
    val pairs = batch.map {
      case (dataFlow, target) ⇒ net.backward(target, dataFlow)
    }
    val batchLoss = pairs.last._1 //todo: confirm on this
    val batchSize = pairs.size
    val paramsGrads = accumulate(pairs.map(_._2))
    val (newParams, newContext, lossInfo) = calculate(paramsGrads, lastResult, batchLoss, batchSize)

    val newLayers = newParams.map {
      case (l, ps) ⇒ l.updateParams(ps)
    }
    val newLayersSorted = net.hiddenLayers.map { oldLayer ⇒
      newLayers.find(_.id == oldLayer.id).getOrElse(oldLayer) //it is possible that some layer didn't get updated and thus use the old layer instead
    }

    val newNet: Trainee = build(net, newLayersSorted)
    BatchTrainer.BatchResult(lossInfo, newNet, batchSize, newContext)
  }

  def accumulate(netParamGradients: Iterable[NetParamGradients]): NetParamGradients =
    netParamGradients.reduce { (npg1, npg2) ⇒
      (npg1, npg2).zipped.map {
        case ((layer1, paramGradients1), (layer2, paramGradients2)) ⇒
          assert(layer1 == layer2) //assume sequence of the two NetParamGradients are the same, but assert the match here
          val newParamGradient = (paramGradients1, paramGradients2).zipped.map { (paramGradient1, paramGradient2) ⇒
            assert(paramGradient1.param == paramGradient2.param) //assume the sequence remain the same, but assert the match here
            paramGradient1.copy(value = paramGradient1.value + paramGradient2.value)
          }
          (layer1, newParamGradient)
      }
    }

  def calculate(paramGrads: NetParamGradients, lastIterationResult: BatchResult, loss: Loss, batchSize: Int): (NetParams, CalculationContext, LossInfo)
}

object BatchTrainer {
  case class LossInfo(l1Decay: Loss, l2Decay: Loss, cost: Loss) {
    val total: Loss = l1Decay + l2Decay + cost
  }

  case class BatchResult[Trainee <: Net, CalculationContext](lossInfo: LossInfo, net: Trainee, batchSize: Int, calcContext: CalculationContext)

}

