package glaux.linalg

import glaux.linalg.Tensor._

// Implement this trait to provide an implementation of the linarg interfaces.
trait Implementation {
  implicit val rBuilder: RowBuilder
  implicit val mBuilder: MatrixBuilder
  implicit val vBuilder: VolBuilder
  implicit val t4Builder: Tensor4Builder
  implicit val genBuilder: GenTensorBuilder[Tensor]
}
