/*
* Code adapted from here: http://codepen.io/miam84/pen/aNMryw
* Original code by BY Yann L
*/
function getMouseOffset (e, id) {
    var xpos, ypos;
    if (typeof e.offsetX === 'undefined') { // ff hack
        xpos = e.pageX - $('#' + id).offset().left; //dans ce cas, jQuery nécessaire pour l'appel d'offset
        ypos = e.pageY - $('#' + id).offset().top;
    } else {
        xpos = e.offsetX;
        ypos = e.offsetY;
    }
    return { x: xpos, y: ypos };
}

/**
* Extends Canvas element to use it as a paint element
*/
var extendCanvas = function (canvas) {
  if (typeof canvas === 'string') {
    canvas = document.getElementById(canvas);
  }
  try {
    canvas.ctx = canvas.getContext('2d');
    canvas.ctx.strokeStyle = "black";
    canvas.ctx.lineWidth = 8;
    canvas.lineJoin = "round";
    canvas.ctx.scale(1, 1);
    canvas.isLocked = false;
    canvas.hasDrawn = false;

    canvas.disableSave = true;
    canvas.pixels = [];
    canvas.cpixels = [];
    canvas.xyLast = {};
    canvas.xyAddLast = {};
    canvas.calculate = false;

    //start drawing a stroke from a position
    canvas.draw = function (x, y) {
      this.ctx.lineTo(x, y);
      this.ctx.stroke();
    };
    //Get the mouse position in the canvas
    canvas.getCursorCoords = function (e) {
      return getMouseOffset(e, this.id);
    };
    //Clear the content of the canvas element
    canvas.clear = function () {
      if (!this.isLocked) {
        this.ctx.clearRect(0, 0, this.width, this.height);
        this.emptyRelatedField();
        this.isModified(false);
      }
    };
    //Lock the canvas element
    canvas.lock = function (shouldLock) {
      this.isLocked = shouldLock;
      if (shouldLock) {
        $(this).addClass('disabled');
      } else {
        $(this).removeClass('disabled');
      }
    };
    canvas.isModified = function (wasModified) {
      if (wasModified) {
        this.fillRelatedField();
        $(this).addClass('modified');
      } else {
        this.emptyRelatedField();
        $(this).removeClass('modified');
      }
    };
    //Empty the field that contains the base64 string
    canvas.emptyRelatedField = function () {
      var relatedField = document.getElementById('data-' + this.id);
      if (relatedField) {
        relatedField.value = "";
      }
    };
    //Fill the filed that contains the base64 string
    canvas.fillRelatedField = function () {
      document.getElementById('data-' + this.id).value = this.toDataURL();
    };
    //Load a passed image
    canvas.loadImage = function (base64Img) {
      //if (!canvas.isLocked) {
      var image = new Image();
      var thisCanvas = this;
      if (base64Img.indexOf("data:image/png;base64,") === -1) {
        base64Img = "data:image/png;base64," + base64Img;
      }
      image.src = base64Img;
      image.onload = function () {
        var offset = { x: 0, y: 0 };
        //center the image in the canvas
        try {
          offset.x = ((thisCanvas.width / 2) - (image.width / 2));
          offset.y = ((thisCanvas.height / 2) - (image.height / 2));
          if (offset.x < 0 || offset.y < 0) {
            throw {
              name: "Painting error",
              message: "Image has a negative offset.",
              toString: function () { return this.name + ": " + this.message; }
            };
          }
        } catch (err) {
          offset = { x: 0, y: 0 };
        }
        thisCanvas.ctx.drawImage(image, offset.x, offset.y);
      };
    };

    /**
            * This method remove the event by default, to load them only when clicking in the canvas
            */
    function remove_event_listeners() {
      canvas.removeEventListener('mousemove', on_mousemove, false);
      canvas.removeEventListener('mouseup', on_mouseup, false);
      canvas.removeEventListener('touchmove', on_mousemove, false);
      canvas.removeEventListener('touchend', on_mouseup, false);
      document.body.removeEventListener('mouseup', on_mouseup, false);
      document.body.removeEventListener('touchend', on_mouseup, false);
    };
    //Event when the mouse is clicked
    function on_mousedown(e) {
      if (!canvas.isLocked) {
        e.preventDefault();
        e.stopPropagation();

        canvas.hasDrawn = false;
        //we activate the mouse and touch events
        canvas.addEventListener('mouseup', on_mouseup, false);
        canvas.addEventListener('mousemove', on_mousemove, false);
        canvas.addEventListener('touchend', on_mouseup, false);
        canvas.addEventListener('touchmove', on_mousemove, false);
        document.body.addEventListener('mouseup', on_mouseup, false);
        document.body.addEventListener('touchend', on_mouseup, false);

        var xy = canvas.getCursorCoords(e);
        canvas.ctx.beginPath();
        canvas.pixels.push('moveStart');
        canvas.ctx.moveTo(xy.x, xy.y);
        canvas.pixels.push(xy.x, xy.y);
        canvas.xyLast = xy;
      }
    };
    //Event when the mouse is moving.
    function on_mousemove(e, finish) {
      if (!canvas.isLocked) {
        e.preventDefault();
        e.stopPropagation();

        canvas.hasDrawn = true;
        var xy = canvas.getCursorCoords(e);
        var xyAdd = {
          x: (canvas.xyLast.x + xy.x) / 2,
          y: (canvas.xyLast.y + xy.y) / 2
        };
        if (canvas.calculate) {
          var xLast = (canvas.xyAddLast.x + canvas.xyLast.x + xyAdd.x) / 3;
          var yLast = (canvas.xyAddLast.y + canvas.xyLast.y + xyAdd.y) / 3;
          canvas.pixels.push(xLast, yLast);
        } else {
          canvas.calculate = true;
        }
        canvas.ctx.quadraticCurveTo(canvas.xyLast.x, canvas.xyLast.y, xyAdd.x, xyAdd.y);
        canvas.pixels.push(xyAdd.x, xyAdd.y);
        canvas.ctx.stroke();
        canvas.ctx.beginPath();
        canvas.ctx.moveTo(xyAdd.x, xyAdd.y);
        canvas.xyAddLast = xyAdd;
        canvas.xyLast = xy;
      }
    };
    //Event when the click is released
    function on_mouseup(e) {
      if (!canvas.isLocked) {
        if (!canvas.hasDrawn) {//If there was no move, draw a single point
          var pos = canvas.getCursorCoords(e);
          canvas.ctx.rect(pos.x, pos.y, 1, 1);
        }
        remove_event_listeners();
        canvas.disableSave = false;
        canvas.ctx.stroke();
        canvas.pixels.push('e');
        canvas.calculate = false;
        canvas.isModified(true);
        canvas.hasDrawn = false;
      }
    };
    //We activate only the click or touch event.
    // ReSharper disable once Html.EventNotResolved correspond à un événement de toucher sur écran tactile
    canvas.addEventListener('touchstart', on_mousedown, false);
    canvas.addEventListener('mousedown', on_mousedown, false);

  } catch (err) {
    //If someting went wrong, notify the user.
    console.error("Canvas not initialized. Painting is not activated.");
  }

  return canvas;
};


var myC = extendCanvas('paint');
$('#clear').on('click', (e) => {
  myC.clear();
  myC.emptyRelatedField();
});
