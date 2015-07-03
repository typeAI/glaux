package glaux.linalg.impl.nd4j

import glaux.linalg.Dimension.{Row, TwoD, ThreeD}
import glaux.linalg._
import glaux.linalg.Tensor._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Implicits {
  implicit val tensorBuilder: CanBuildFrom[INDArray, Tensor]  = indArray =>
    Dimension.of(indArray.shape) match {
      case d @ ThreeD(_,_,_) => VolImp(indArray)
      case d @ TwoD(_,_) => MatrixImp(indArray)
      case d @ Row(_) => RowVectorImp(indArray)
    }

  type IndArrayVolBuilder[V <: Tensor] = CanBuildFrom[INDArray, V]

  implicit val indArrayRowBuilder: IndArrayVolBuilder[RowVector] = RowVectorImp.apply
  implicit val indArrayMatrixBuilder: IndArrayVolBuilder[Matrix] = MatrixImp.apply
  implicit val indArrayVol3DBuilder: IndArrayVolBuilder[Vol] = VolImp.apply


  implicit def builder[V <: Tensor](implicit indArrayVolBuilder: IndArrayVolBuilder[V]): TensorBuilder[V] = (createINDArray _).tupled.andThen(indArrayVolBuilder)
  implicit val rBuilder: RowBuilder = builder[RowVector]
  implicit val mBuilder: MatrixBuilder = builder[Matrix]
  implicit val vBuilder: Vol3DBuilder = builder[Vol]
  implicit val genBuilder: GenTensorBuilder[Tensor] = (createINDArray _).tupled.andThen(tensorBuilder)

  implicit def toWithIndArray(v: Tensor): WithIndArray = new WithIndArray{ val indArray = v.asInstanceOf[ND4JBackedTensor].indArray }

  private def createINDArray(dimension: Dimension, data: Seq[Double]): INDArray = {
    assert(dimension.totalSize == data.length, s"data length ${data.length} does not conform to $dimension" )
    Nd4j.create(data.toArray, dimension.shape)
  }

}

class ND4JImplementation extends Implementation {
  implicit val rBuilder = Implicits.rBuilder
  implicit val mBuilder = Implicits.mBuilder
  implicit val vBuilder = Implicits.vBuilder
  implicit val genBuilder = Implicits.genBuilder
}



