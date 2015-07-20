package glaux.linalg.impl.nd4j

import glaux.linalg.Dimension._
import glaux.linalg._
import org.nd4j.linalg.api.ndarray.INDArray


trait TensorImpl extends Tensor with TensorOperationsImpl with WithIndArray {
  def getSalar(indices: Seq[Int]): Double = indArray.getDouble(indices:_*)
}

trait WithIndArray {
  val indArray: INDArray
}


trait TensorOperationsImpl extends TensorOperations {
  this: WithIndArray =>

  import Implicits.tensorBuilder
  import Implicits.toWithIndArray
  def +(that: Tensor): Tensor = this.indArray.add(that.indArray)

  def -(that: Tensor): Tensor = indArray.sub(that.indArray)

  /** element-by-element multiplication */
  def *(that: Tensor): Tensor = indArray.mul(that.indArray)

  /** matrix multiplication */
  def **(that: Tensor): Tensor = indArray.mmul(that.indArray)

  def /(that: Tensor): Tensor = indArray.div(that.indArray)

  /** right division ... is this the correct symbol? */
  def \(that: Tensor): Tensor = indArray.rdiv(that.indArray)

  def +(that: Number): Tensor = indArray.add(that)
  def -(that: Number): Tensor = indArray.sub(that)
  def *(that: Number): Tensor = indArray.mul(that)
  def /(that: Number): Tensor = indArray.div(that)
  def \(that: Number): Tensor = indArray.rdiv(that)

  def T: Tensor = indArray.transpose

}


protected sealed abstract class ND4JBackedTensor(val indArray: INDArray) extends TensorImpl {
  val dimensionFactory: DimensionFactory[Dimensionality]

  lazy val dimension: Dimensionality = dimensionFactory.create(indArray.shape())
  def seqView: Seq[Double] = {

    new Seq[Double] {
      outer =>
      def length = dimension.totalSize
      def apply(idx: Int) = indArray.getDouble(idx)
      //todo: although it's trivial to implement, I think there should be an existing iterator for seq somewhere.
      def iterator = new Iterator[Double] {
        var index: Int = 0
        override def hasNext: Boolean = index < outer.length
        override def next(): Double = {
          index += 1
          apply(index - 1)
        }
      }
    }
  }

}

case class RowVectorImp(override val indArray: INDArray)  extends ND4JBackedTensor(indArray) with RowVector {
  val dimensionFactory = Row
  def apply(index: Int): Double = indArray.getDouble(index)
}

case class MatrixImp(override val indArray: INDArray)     extends ND4JBackedTensor(indArray) with Matrix {
  val dimensionFactory = TwoD
}

case class VolImp(override val indArray: INDArray)      extends ND4JBackedTensor(indArray) with Vol {
  val dimensionFactory = ThreeD
}

case class Tensor4Imp(override val indArray: INDArray)      extends ND4JBackedTensor(indArray) with Tensor4 {
  val dimensionFactory = FourD
}

