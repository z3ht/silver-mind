const target = "https://localhost:8421";


function onPlayerMove (oldPos, newPos) {
  $.post(`${target}/chess/move?Move=${oldPos}${newPos}`, function(data, status){
    if (data === "false") {
      return;
    }
    board.position(data);
  });
}


function doComputerMove () {
  $.post(`${target}/chess/next`, function(data, status){
    board.position(data);
  });
}


function undo () {
  $.post(`${target}/chess/undo`, function(data, status){
    board.position(data);
  });
}


function start () {
  $.post(`${target}/chess/start`, function(data, status){
    board.position('start');
  });
}


var config = {
  draggable: true,
  onMoveEnd: onPlayerMove
}

var board = ChessBoard('board', config);

$('#start').on('click', start);
$('#next').on('click', doComputerMove);
$('#undo').on('click', undo);
