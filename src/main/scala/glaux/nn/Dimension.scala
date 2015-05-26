package glaux.nn

import org.nd4j.linalg.api.ndarray.INDArray

trait Dimension

object Dimension {
  type Shape = Array[Int]

  trait DimensionFactory[DT <: Dimension] {
    val createFunction: PartialFunction[Shape, DT]
    lazy val tryCreate: Shape => Option[DT] = createFunction.lift
    def create(shape: Shape): DT = tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  def unsupported(shape: Shape):Exception = new UnsupportedOperationException(s"unrecognized matrix shape $shape")

  def ofSpecific[DT <: Dimension](matrix: INDArray)(implicit df: DimensionFactory[DT]): DT = {
    val shape = matrix.shape()
    df.tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  def of(matrix: INDArray): Dimension = {
    val shape = matrix.shape()
    val df3d = implicitly[DimensionFactory[Dimension3D]]
    val df1d = implicitly[DimensionFactory[DimensionArray]]
    df1d.tryCreate(shape).orElse(df3d.tryCreate(shape)).getOrElse(throw unsupported(shape))
  }

  case class Dimension3D(x: Int, y: Int, z: Int) extends Dimension {
    assert(x > 0 && y > 0 && z > 0)
  }

  case class DimensionArray(size: Int) extends Dimension {
    assert(size > 0)
  }

  implicit object DimensionArray extends DimensionFactory[DimensionArray] {
    val dimIndexOfData: Int = 1

    val createFunction: PartialFunction[Shape, DimensionArray] = {
      case Array(1, size) => DimensionArray(size)
    }
  }

  implicit object Dimension3D extends DimensionFactory[Dimension3D] {
    val createFunction = PartialFunction[Shape, Dimension3D] {
      case Array(x, y, z) => Dimension3D(x,y,z)
    }
  }

}
