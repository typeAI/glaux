package glaux.linalg.back.nd4j

import glaux.linalg.Dimension.{Row, TwoD, ThreeD}
import glaux.linalg._
import glaux.linalg.Vol._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Implicits {
  implicit val volBuilder: CanBuildFrom[INDArray, Vol]  = indArray =>
    Dimension.of(indArray.shape) match {
      case d @ ThreeD(_,_,_) => Vol3DImp(indArray)
      case d @ TwoD(_,_) => MatrixImp(indArray)
      case d @ Row(_) => RowVectorImp(indArray)
    }

  type IndArrayVolBuilder[V <: Vol] = CanBuildFrom[INDArray, V]

  implicit val indArrayRowBuilder: IndArrayVolBuilder[RowVector] = RowVectorImp.apply
  implicit val indArrayMatrixBuilder: IndArrayVolBuilder[Matrix] = MatrixImp.apply
  implicit val indArrayVol3DBuilder: IndArrayVolBuilder[Vol3D] = Vol3DImp.apply


  implicit def builder[V <: Vol](implicit indArrayVolBuilder: IndArrayVolBuilder[V]): VolBuilder[V] = (createINDArray _).tupled.andThen(indArrayVolBuilder)
  implicit val rBuilder: RowBuilder = builder[RowVector]
  implicit val mBuilder: MatrixBuilder = builder[Matrix]
  implicit val vBuilder: Vol3DBuilder = builder[Vol3D]
  implicit val genBuilder: GenVolBuilder[Vol] = (createINDArray _).tupled.andThen(volBuilder)

  implicit def toWithIndArray(v: Vol): WithIndArray = new WithIndArray{ val indArray = v.asInstanceOf[ND4JBackedVol].indArray }

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



