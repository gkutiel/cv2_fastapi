<script lang="ts">
  $: frameSrc = "";
  $: faceSrc = "";

  new WebSocket("ws://localhost:8000/ws").onmessage = (event) => {
    const data = JSON.parse(event.data);
    const { frame_src, face_src, ear_left, ear_right, trans, rot } = data;

    frameSrc = frame_src;
    faceSrc = face_src;

    //   if (!crop) {
    //     crop_el.src =
    //       "https://theawesomedaily.com/wp-content/uploads/2017/05/funny-faces-101.jpeg";
    //   } else {
    //     crop_el.src = `data:image/jpg;base64,${crop}`;
    //   }

    //   const [trans_x, trans_y, trans_z] = trans;
    //   set("trans_x", trans_x);
    //   set("trans_y", trans_y);
    //   set("trans_z", trans_z);

    //   const [rot_x, rot_y, rot_z] = rot;
    //   set("rot_x", `${rot_x}°`);
    //   set("rot_y", `${rot_y}°`);
    //   set("rot_z", `${rot_z}°`);
    //   set("left_ear", ear_left);
    //   set("right_ear", ear_right);
  };
</script>

<main>
  <div class="row">
    <img class="vid" alt="" src={`data:image/jpg;base64,${frameSrc}`} />
    <img class="vid" alt="" src={`data:image/jpg;base64,${faceSrc}`} />
  </div>
</main>

<!-- <div class="info">
  <table>
    <tr>
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
    <tr>
      <td>Translation</td>
      <td id="trans_x"></td>
      <td id="trans_y"></td>
      <td id="trans_z"></td>
    </tr>
  </table>
</div>

<div class="info">
  <table>
    <tr>
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
    <tr>
      <td>Rotation</td>
      <td id="rot_x"></td>
      <td id="rot_y"></td>
      <td id="rot_z"></td>
    </tr>
  </table>

  <div id="ears">
    <div>
      Left EAR
      <p id="left_ear"></p>
    </div>
    <div>
      Right EAR
      <p id="right_ear"></p>
    </div>
  </div>
</div> -->

<style>
  img.vid {
    height: 512px;
  }
</style>
